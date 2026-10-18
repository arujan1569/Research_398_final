import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas.prediction import ModelId, ModelInfo, PredictionResponse
from app.services.model_service import ModelUnavailableError
from app.utils.image_processing import ImageValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api', tags=['Rice leaf classification'])


@router.get('/health')
def health(request: Request):
    service = request.app.state.model_service
    available = {key: info.available for key, info in service.info.items()}
    ready = len(available) == 2 and all(available.values())
    return {'status': 'ok' if ready else 'degraded', 'models': available,
            'tensorflow_version': service.tensorflow_version,
            'max_upload_bytes': request.app.state.settings.max_upload_bytes}


@router.get('/models', response_model=list[ModelInfo])
def models(request: Request):
    return list(request.app.state.model_service.info.values())


@router.post('/predict', response_model=PredictionResponse)
async def predict(request: Request, file: Annotated[UploadFile, File(description='JPEG, PNG or WebP rice leaf image')],
                  model: Annotated[ModelId, Form()]):
    try:
        limit = request.app.state.settings.max_upload_bytes
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f'Image exceeds the {limit // (1024*1024)} MB upload limit.')
        return await run_in_threadpool(request.app.state.model_service.predict, model, data)
    except ImageValidationError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Inference failed for %s', model)
        raise HTTPException(500, 'Inference could not be completed. Please retry or check the server logs.') from exc
    finally:
        await file.close()
