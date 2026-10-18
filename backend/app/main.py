from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from app.config import Settings, settings
from app.routers.prediction import router
from app.services.model_service import ModelService


class RequestSizeLimit:
    """Bound the multipart body before disk spooling, including chunked uploads."""
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in {'POST', 'PUT', 'PATCH'}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await JSONResponse({'detail': 'Invalid Content-Length.'}, status_code=400)(scope, receive, send)
        if length > self.limit:
            return await JSONResponse({'detail': 'Request exceeds the upload limit.'}, status_code=413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > self.limit:
                raise HTTPException(413, 'Request exceeds the upload limit.')
            return message

        await self.app(scope, limited_receive, send)


def create_app(config: Settings = settings, service: ModelService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = config
        app.state.model_service = service or ModelService(config)
        if service is None:
            app.state.model_service.load()
        yield

    app = FastAPI(title='OryzaAI · Rice Leaf Classification', version='1.0.0', lifespan=lifespan,
                  description='Two independently trained six-class models. Scores are uncalibrated; images are not retained.')
    app.add_middleware(RequestSizeLimit, limit=config.max_upload_bytes + 64 * 1024)
    app.add_middleware(CORSMiddleware, allow_origins=list(config.cors_origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
    app.include_router(router)
    # A production React build is served by the same FastAPI process.
    dist = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
    return app


app = create_app()
