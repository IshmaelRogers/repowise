"""FastAPI application factory for the repowise server.

The ``create_app()`` function builds and configures the FastAPI instance.
The ``lifespan`` context manager handles startup (DB, FTS, vector store,
scheduler) and shutdown (cleanup).
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi import FastAPI, Request, Response
from repowise.core.persistence.database import (
    create_engine,
    create_session_factory,
    get_session,
    init_db,
    resolve_db_url,
)
from repowise.core.persistence.search import FullTextSearch
from repowise.core.persistence.vector_store import InMemoryVectorStore
from repowise.core.providers.embedding.base import MockEmbedder
from repowise.server import __version__
from repowise.server.routers import (
    blast_radius,
    chat,
    claude_md,
    costs,
    dead_code,
    decisions,
    git,
    graph,
    health,
    jobs,
    knowledge_map,
    pages,
    providers,
    repos,
    search,
    security,
    symbols,
    webhooks,
)
from repowise.server.scheduler import setup_scheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-IP rate limiting (sliding window, in-process)
# ---------------------------------------------------------------------------
_RATE_LIMIT_REQUESTS = int(os.environ.get("REPOWISE_RATE_LIMIT_REQUESTS", "300"))
_RATE_LIMIT_WINDOW = int(os.environ.get("REPOWISE_RATE_LIMIT_WINDOW", "60"))

# Maps client IP → list of request timestamps within the current window
_rate_limit_hits: dict[str, list[float]] = defaultdict(list)


async def _rate_limit_middleware(request: Request, call_next):
    """Reject requests from IPs that exceed the configured sliding-window limit.

    Defaults to 300 requests per 60 seconds per client IP.  Override with
    REPOWISE_RATE_LIMIT_REQUESTS and REPOWISE_RATE_LIMIT_WINDOW env vars.
    Disabled when REPOWISE_RATE_LIMIT_REQUESTS is set to 0.
    """
    if _RATE_LIMIT_REQUESTS == 0:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - _RATE_LIMIT_WINDOW

    hits = _rate_limit_hits[client_ip]
    # Prune expired timestamps
    hits = [t for t in hits if t > cutoff]
    _rate_limit_hits[client_ip] = hits
    if len(hits) >= _RATE_LIMIT_REQUESTS:
        return Response(
            content='{"detail":"Too many requests"}',
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
        )
    hits.append(now)
    _rate_limit_hits[client_ip] = hits
    return await call_next(request)


def _build_embedder():
    """Build an embedder from REPOWISE_EMBEDDER env var (default: mock).

    Supported values:
        mock    — deterministic 8-dim SHA-256 embedder (default, no API key needed)
        gemini  — GeminiEmbedder via GEMINI_API_KEY / GOOGLE_API_KEY env var
        openai  — OpenAIEmbedder via OPENAI_API_KEY env var
    """
    name = os.environ.get("REPOWISE_EMBEDDER", "mock").lower()
    if name == "gemini":
        from repowise.core.providers.embedding.gemini import GeminiEmbedder

        dims = int(os.environ.get("REPOWISE_EMBEDDING_DIMS", "768"))
        return GeminiEmbedder(output_dimensionality=dims)
    if name == "openai":
        from repowise.core.providers.embedding.openai import OpenAIEmbedder

        model = os.environ.get("REPOWISE_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbedder(model=model)
    logger.warning("embedder.mock_active — set REPOWISE_EMBEDDER=gemini or openai for real RAG")
    return MockEmbedder()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: create DB engine, session factory, FTS, vector store, scheduler.
    Shutdown: dispose engine, stop scheduler, close vector store.
    """
    # Database
    db_url = resolve_db_url()
    engine = create_engine(db_url)
    await init_db(engine)
    session_factory = create_session_factory(engine)

    # Reset any jobs left in "running" state from a previous server instance
    # (crash or restart) — they can never complete now.
    # Note: with multi-worker deployments this is a best-effort race; the
    # try/except prevents a SQLite lock error from crashing startup.
    try:
        from sqlalchemy import update as sa_update
        from repowise.core.persistence.models import GenerationJob
        from datetime import datetime, UTC as _UTC

        async with get_session(session_factory) as session:
            stale_result = await session.execute(
                sa_update(GenerationJob)
                .where(GenerationJob.status == "running")
                .values(
                    status="failed",
                    error_message="Server restarted — job interrupted",
                    finished_at=datetime.now(_UTC),
                )
            )
            if stale_result.rowcount:
                logger.warning("reset_stale_jobs", extra={"count": stale_result.rowcount})
    except Exception as exc:
        logger.warning("stale_job_reset_failed", extra={"error": str(exc)})

    # Full-text search
    fts = FullTextSearch(engine)
    await fts.ensure_index()

    # Vector store (InMemory default; LanceDB/pgvector configured via env)
    embedder = _build_embedder()
    vector_store = InMemoryVectorStore(embedder=embedder)

    # Background scheduler
    scheduler = setup_scheduler(session_factory)
    scheduler.start()

    # Store on app state
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.fts = fts
    app.state.vector_store = vector_store
    app.state.scheduler = scheduler
    app.state.background_tasks: set = set()  # Strong refs to prevent GC of asyncio tasks

    # Initialize chat tool state (bridges FastAPI state to MCP tool globals)
    from repowise.server.chat_tools import init_tool_state

    init_tool_state(
        session_factory=session_factory,
        fts=fts,
        vector_store=vector_store,
    )

    logger.info("repowise_server_started", extra={"version": __version__})
    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    await vector_store.close()
    await engine.dispose()
    logger.info("repowise_server_stopped")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="repowise API",
        description="REST API for repowise — codebase documentation engine",
        version=__version__,
        lifespan=lifespan,
    )

    # Rate limiting — must be added before CORS so OPTIONS pre-flights also count
    app.middleware("http")(_rate_limit_middleware)

    # CORS — configurable via REPOWISE_ALLOWED_ORIGINS (comma-separated list).
    # When using a wildcard origin, credentials cannot be included per the
    # browser spec, so allow_credentials is only enabled for explicit origins.
    _allowed_origins_env = os.environ.get("REPOWISE_ALLOWED_ORIGINS", "")
    _allowed_origins: list[str] = (
        [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
        if _allowed_origins_env
        else ["*"]
    )
    _allow_credentials = _allowed_origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(LookupError)
    async def not_found_handler(request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def bad_request_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # Include routers
    app.include_router(health.router)
    app.include_router(repos.router)
    app.include_router(pages.router)
    app.include_router(search.router)
    app.include_router(jobs.router)
    app.include_router(symbols.router)
    app.include_router(graph.router)
    app.include_router(webhooks.router)
    app.include_router(git.router)
    app.include_router(dead_code.router)
    app.include_router(claude_md.router)
    app.include_router(decisions.router)
    app.include_router(chat.router)
    app.include_router(providers.router)
    app.include_router(costs.router)
    app.include_router(security.router)
    app.include_router(blast_radius.router)
    app.include_router(knowledge_map.router)

    return app
