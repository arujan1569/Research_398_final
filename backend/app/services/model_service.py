"""Load trusted local exports once and run independent model inference."""
import hashlib
import csv
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.schemas.prediction import ModelInfo, PredictionResponse
from app.utils.image_processing import prepare_image

logger = logging.getLogger(__name__)
MODEL_SPECS = {
    'simple_cnn': ('Simple CNN', 'Trained from scratch · 3 convolution blocks', 'simple_cnn'),
    'mobilenetv3': ('MobileNetV3Large', 'ImageNet transfer learning · fine-tuned backbone', 'keras'),
}
SCORE_NOTE = ('Uncalibrated six-class model scores, not diagnostic probabilities. '
              'Unsupported conditions and non-rice images can receive high scores. '
              'Research use; confirm findings with an agricultural specialist.')


class ModelUnavailableError(RuntimeError):
    pass


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = {}
        self.info: dict[str, ModelInfo] = {}
        self.locks = {key: threading.Lock() for key in MODEL_SPECS}
        self.tensorflow_version: str | None = None

    def load(self) -> None:
        # A missing or incompatible export does not stop the other model working.
        for identifier, (name, description, metric_key) in MODEL_SPECS.items():
            info = ModelInfo(id=identifier, name=name, description=description,
                             available=False, class_names=[], input_shape=[1, 224, 224, 3], preprocessing={})
            self.info[identifier] = info
            folder = self.settings.model_dir / identifier
            try:
                labels = json.loads((folder / 'class_names.json').read_text(encoding='utf-8'))
                config = json.loads((folder / 'preprocessing.json').read_text(encoding='utf-8'))
                if not isinstance(labels, list) or len(labels) != 6 or len(set(labels)) != 6 or not all(isinstance(x, str) and x for x in labels):
                    raise ValueError('Expected six distinct class labels.')
                if config.get('input_shape') != [1, 224, 224, 3]:
                    raise ValueError('Unsupported input dimensions.')
                if config.get('input_range', config.get('float_input_range')) != [0, 255]:
                    raise ValueError('Unsupported pixel range.')
                if 'inside' not in config.get('normalization', '').lower() or 'bilinear' not in config.get('resize', '').lower():
                    raise ValueError('Preprocessing does not match the supplied exports.')
                info.class_names = labels
                info.preprocessing = config
                path = folder / 'model.keras'
                info.size_bytes = path.stat().st_size
                info.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                metrics_path = folder / 'reports' / 'test_metrics.json'
                if metrics_path.exists():
                    info.metrics = json.loads(metrics_path.read_text())[metric_key]
                split_path = folder / 'reports' / 'split_manifest.csv'
                if split_path.exists():
                    with split_path.open(encoding='utf-8', newline='') as stream:
                        identities = sorted((r['pixel_sha256'], r['label'], r['split']) for r in csv.DictReader(stream))
                    info.split_fingerprint = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
                import tensorflow as tf
                self.tensorflow_version = tf.__version__
                model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
                if tuple(model.input_shape[1:]) != (224, 224, 3) or tuple(model.output_shape[1:]) != (len(labels),):
                    raise ValueError('Model tensors do not match the metadata.')
                info.parameter_count = int(model.count_params())
                # Warm up so the first user does not pay graph initialization cost.
                self._scores(model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False), len(labels))
                self.models[identifier] = model
                info.available = True
                logger.info('Loaded %s (%s parameters)', identifier, info.parameter_count)
            except Exception:
                logger.exception('Could not load model %s', identifier)
                info.status_message = 'Model unavailable. Check the local model package and server logs.'

    @staticmethod
    def _scores(output, count: int) -> np.ndarray:
        values = np.asarray(output, dtype=np.float32)
        if values.shape != (1, count) or not np.isfinite(values).all():
            raise RuntimeError('Invalid inference output.')
        scores = values[0]
        if (scores < 0).any() or (scores > 1).any() or not np.isclose(scores.sum(), 1, atol=1e-3):
            raise RuntimeError('Expected softmax scores.')
        return scores

    def predict(self, identifier: str, data: bytes) -> PredictionResponse:
        if identifier not in self.models:
            raise ModelUnavailableError('Selected model is unavailable. Check API status and try again.')
        started = time.perf_counter()
        info = self.info[identifier]
        tensor, image = prepare_image(data, max_pixels=self.settings.max_image_pixels, input_shape=info.input_shape)
        prepared = time.perf_counter()
        with self.locks[identifier]:
            inference_start = time.perf_counter()
            scores = self._scores(self.models[identifier](tensor, training=False), len(info.class_names))
            finished = time.perf_counter()
        order = np.argsort(-scores, kind='stable')
        ranking = [{'class_name': info.class_names[int(i)], 'score': float(scores[i])} for i in order]
        return PredictionResponse(
            request_id=uuid.uuid4().hex[:12], created_at=datetime.now(timezone.utc).isoformat(),
            model=identifier, model_name=info.name, predicted_class=ranking[0]['class_name'],
            prediction_score=ranking[0]['score'], top_3_predictions=ranking[:3], all_predictions=ranking,
            image=image, input_shape=info.input_shape, model_sha256=info.sha256,
            timings={'preprocessing_ms': round((prepared-started)*1000, 2),
                     'inference_ms': round((finished-inference_start)*1000, 2),
                     'total_ms': round((finished-started)*1000, 2)}, note=SCORE_NOTE,
        )
