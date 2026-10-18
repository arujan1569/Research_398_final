# OryzaAI FastAPI backend

Python 3.12, FastAPI, TensorFlow 2.20.0, Keras 3.11.3. Both supplied trained models are installed under `models/`; no retraining is performed.
hhhh
## Structure

```text
backend/
  app/
    __init__.py
    main.py                  # lifespan, CORS, body limits, optional React hosting
    config.py                # .env settings
    routers/prediction.py    # API endpoints
    services/model_service.py
    schemas/prediction.py
    utils/image_processing.py
  models/
    simple_cnn/
      model.keras
      class_names.json
      preprocessing.json
    mobilenetv3/
      model.keras
      class_names.json
      preprocessing.json
  tests/test_prediction.py
  .env.example
  .gitignore
  requirements.txt
  requirements-dev.txt
```

Each Python package includes `__init__.py`. Additional model metadata includes the original model card, preprocessing reference, source archive checksum, experiment configuration, and test reports. Original `.tflite` exports are retained; this API uses `.keras` only.

## Install and start

From `backend/` in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/docs for Swagger, or `/redoc` for reference documentation. TensorFlow runs on CPU in this native Windows environment. Dependencies reproduce the notebook inference versions; see [TensorFlow installation](https://www.tensorflow.org/install/pip) and [Keras loading documentation](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/).

`CORS_ORIGINS` is a comma-separated exact allowlist. `MODEL_DIR` resolves relative to `backend/` unless absolute. `MAX_UPLOAD_MB` defaults to 15; `MAX_IMAGE_PIXELS` defaults to 25 million. The server loads `.env` automatically. Bind host and port using Uvicorn flags. Keep a single worker for local use: each extra process loads another copy of both models.

## Endpoints

| Method | URL | Behavior |
|---|---|---|
| GET | `/api/health` | Liveness, model availability, TensorFlow version, upload limit. `status: degraded` means at least one model is unavailable. |
| GET | `/api/models` | Exact class order, preprocessing, actual parameter counts, model SHA256, exported test metrics. |
| POST | `/api/predict` | Multipart `file` plus `model`: `simple_cnn` or `mobilenetv3`. Only the selected model runs. |

```powershell
curl.exe http://127.0.0.1:8001/api/health
curl.exe http://127.0.0.1:8001/api/models
curl.exe -X POST http://127.0.0.1:8001/api/predict -F "model=simple_cnn" -F "file=@C:/path/to/leaf.jpg"
```

The full response schema is in Swagger. It includes `predicted_class`, a numeric `prediction_score` in [0,1], `top_3_predictions` (`class_name`, `score`), all six ranked scores, source image dimensions, model checksum, request ID, UTC timestamp, and measured preprocessing/inference/total milliseconds. Total processing includes any wait for the selected model's lock; inference timing excludes queueing. It is not a network round-trip measurement.

Example response excerpt from an actual synthetic green-image smoke test (not a disease accuracy example):

```json
{
  "model": "simple_cnn",
  "model_name": "Simple CNN",
  "predicted_class": "Healthy Rice Leaf",
  "prediction_score": 0.9891467094421387,
  "top_3_predictions": [
    {"class_name": "Healthy Rice Leaf", "score": 0.9891467094421387},
    {"class_name": "Leaf scald", "score": 0.0075339386239647865},
    {"class_name": "Brown Spot", "score": 0.0017473491607233882}
  ],
  "input_shape": [1, 224, 224, 3]
}
```

The complete generated response is saved locally in `../artifacts/prediction-example.json`. The high score on a synthetic image illustrates why the UI labels scores as uncalibrated and does not claim to reject non-rice inputs.

Errors use `detail`: 400 empty/corrupted/animated image, 413 upload/pixel/request limit, 415 unsupported actual image format, 422 missing/invalid form field, 503 model unavailable, 500 unexpected inference failure. Validation errors may use FastAPI's structured detail array. Internal exceptions are logged only on the server. A partial model load failure leaves the other model usable.

## Preprocessing contract

Both provided metadata files require float32 `[1,224,224,3]` in `[0,255]`. Apply EXIF transpose; composite transparency onto black; RGB conversion; `ImageOps.contain` with Pillow bilinear resizing; center on a black 224×224 canvas using integer floor offsets. Simple CNN includes `Rescaling(1/255)` and MobileNetV3 includes its own normalization. Do not divide by 255 or call an application preprocessor externally. Resizing is letterboxing, not cropping or stretching.

Class order is read separately for each model. New incompatible preprocessing metadata is rejected rather than guessed. Models are loaded once in the lifespan with `compile=False`, `safe_mode=True`, shape checks, and warmup. Only trusted local model packages should be installed. Output must be a finite six-class softmax vector. No fabricated fallback predictions exist.

Upload filename and MIME type are not trusted: Pillow validates actual content. Animated images are rejected. Compressed body size is capped before multipart parsing (64 KiB allowance for form headers), and a separate pixel limit prevents oversized decoded images. Uploads are closed after every request; no application image storage or prediction database is used. The multipart parser may temporarily spool larger uploads to the OS temporary directory and removes them on close.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Tests load both real trained models. They cover JPEG/PNG/WebP inference, exact agreement with direct model output, exact tensor parity with each exported reference for RGB/alpha/EXIF/grayscale, invalid and truncated files, animated images, pixel/body/file limits (including chunked requests), model availability, sanitization, and CORS.

The original `models/*/predict.py` files are preserved as provenance and preprocessing references; their original filename assumptions differ from the standardized `model.keras`. Use this API for predictions.

## Research interpretation

Scores are uncalibrated closed-set outputs, not diagnostic probabilities. Non-rice images and unsupported conditions can receive high scores. Neither model returns lesion boxes, segmentation, disease severity, calibration estimates, or validated unknown-image rejection. Metrics come from internal test reports, not field validation. Verify split manifests before presenting cross-model results as a controlled comparison.
