import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cashwatch.analyzers import Analyzer, build_analyzer
from cashwatch.api import api_router, ui_router
from cashwatch.config import Settings
from cashwatch.jobs import JobManager
from cashwatch.rules import load_registry


def create_app(settings: Settings | None = None, analyzer: Analyzer | None = None) -> FastAPI:
    settings = settings or Settings()
    registry = load_registry()
    manager = JobManager(settings, analyzer or build_analyzer(settings), registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        check = getattr(manager.analyzer, "check", None)
        app.state.analyzer_status = check() if check else (True, manager.analyzer.name)
        if not app.state.analyzer_status[0]:
            logging.getLogger("cashwatch").warning("analyzer not ready: %s", app.state.analyzer_status[1])
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="cashwatch", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.manager = manager
    app.state.registry = registry
    app.include_router(api_router)

    if settings.ui_enabled:
        static_dir = resources.files("cashwatch").joinpath("static")
        app.include_router(ui_router)
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(str(static_dir.joinpath("index.html")))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return app
