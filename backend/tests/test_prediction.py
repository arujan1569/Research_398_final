"""Integration tests use the actual trained exports, never mock predictions."""
import io
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings, settings
from app.main import create_app
from app.services.model_service import ModelService
from app.utils.image_processing import prepare_image


@pytest.fixture(scope='session')
def service():
    loaded = ModelService(settings)
    loaded.load()
    assert all(m.available for m in loaded.info.values()), 'Both supplied real models must load.'
    return loaded


@pytest.fixture(scope='session')
def client(service):
    with TestClient(create_app(settings, service)) as test_client:
        yield test_client


def image_bytes(fmt='PNG', mode='RGB', size=(311, 173)):
    buffer = io.BytesIO()
    Image.new(mode, size, (46, 134, 67, 128) if mode == 'RGBA' else (46, 134, 67)).save(buffer, format=fmt)
    return buffer.getvalue()


def test_health_and_metadata(client):
    health = client.get('/api/health').json()
    assert health['status'] == 'ok'
    assert health['models'] == {'simple_cnn': True, 'mobilenetv3': True}
    models = client.get('/api/models').json()
    assert [m['parameter_count'] for m in models] == [110534, 3002118]
    assert all(len(m['class_names']) == 6 and len(m['sha256']) == 64 for m in models)
    assert all(m['metrics']['images'] == 734 for m in models)
    assert models[0]['split_fingerprint'] == models[1]['split_fingerprint']
    assert len(models[0]['split_fingerprint']) == 64


@pytest.mark.parametrize('model', ['simple_cnn', 'mobilenetv3'])
@pytest.mark.parametrize('fmt', ['PNG', 'JPEG', 'WEBP'])
def test_real_predictions(client, service, model, fmt):
    data = image_bytes(fmt)
    response = client.post('/api/predict', data={'model': model}, files={'file': ('leaf.img', data, 'application/octet-stream')})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['model'] == model
    assert len(result['all_predictions']) == 6
    assert result['top_3_predictions'] == result['all_predictions'][:3]
    assert result['predicted_class'] == result['top_3_predictions'][0]['class_name']
    assert result['prediction_score'] == result['top_3_predictions'][0]['score']
    assert abs(sum(p['score'] for p in result['all_predictions']) - 1) < 1e-4
    assert result['image']['width'] == 311
    assert result['image']['height'] == 173
    assert result['timings']['inference_ms'] > 0
    tensor, _ = prepare_image(data, max_pixels=settings.max_image_pixels, input_shape=[1,224,224,3])
    expected = np.asarray(service.models[model](tensor, training=False))[0]
    actual = {p['class_name']: p['score'] for p in result['all_predictions']}
    np.testing.assert_allclose([actual[c] for c in service.info[model].class_names], expected, atol=1e-6)


@pytest.mark.parametrize('model', ['simple_cnn', 'mobilenetv3'])
@pytest.mark.parametrize('variant', ['rgb', 'transparent', 'exif', 'grayscale'])
def test_preprocessing_exactly_matches_export_reference(tmp_path, model, variant):
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (173, 311, 3), dtype=np.uint8))
    fmt = 'PNG'
    kwargs = {}
    if variant == 'transparent':
        image = image.convert('RGBA')
        image.putalpha(96)
    elif variant == 'exif':
        fmt = 'JPEG'
        exif = image.getexif()
        exif[274] = 6
        kwargs['exif'] = exif
    elif variant == 'grayscale':
        image = image.convert('L')
    path = tmp_path / f'{variant}.{fmt.lower()}'
    image.save(path, format=fmt, **kwargs)
    reference = runpy.run_path(str(settings.model_dir / model / 'predict.py'))['prepare_image']
    expected = reference(path)
    actual, info = prepare_image(path.read_bytes(), max_pixels=settings.max_image_pixels, input_shape=[1,224,224,3])
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual.max() > 1  # Catch accidental double normalization.
    assert actual.shape == (1,224,224,3)
    if variant == 'exif':
        assert (info['width'], info['height']) == (173,311)


@pytest.mark.parametrize('payload,status', [(b'',400),(b'not an image',400),(image_bytes('BMP'),415),(image_bytes()[:40],400)], ids=['empty','not-an-image','unsupported-bmp','truncated-png'])
def test_invalid_content(client, payload, status):
    response = client.post('/api/predict', data={'model':'simple_cnn'}, files={'file':('pretend.jpg',payload,'image/jpeg')})
    assert response.status_code == status
    assert isinstance(response.json()['detail'], str)


def test_animated_image_rejected(client):
    buffer = io.BytesIO()
    Image.new('RGB',(20,20),'red').save(buffer,format='PNG',save_all=True,append_images=[Image.new('RGB',(20,20),'green')],duration=100,loop=0)
    response = client.post('/api/predict',data={'model':'simple_cnn'},files={'file':('animated.png',buffer.getvalue())})
    assert response.status_code == 400


@pytest.mark.parametrize('data,files', [({'model':'wrong'},{'file':('x.png',image_bytes())}), ({},{'file':('x.png',image_bytes())}), ({'model':'simple_cnn'},None)])
def test_required_fields_and_model(client,data,files):
    assert client.post('/api/predict',data=data,files=files).status_code == 422


def test_request_limit_and_file_limit(service):
    config = Settings(max_upload_bytes=128)
    with TestClient(create_app(config,service)) as limited:
        response = limited.post('/api/predict', data={'model':'simple_cnn'},files={'file':('large.png',image_bytes())})
        assert response.status_code == 413
        response = limited.post('/api/predict',content=b'x'*70000,headers={'Content-Type':'application/octet-stream'})
        assert response.status_code == 413
        boundary = 'test-boundary'
        chunks = [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="large.png"\r\nContent-Type: image/png\r\n\r\n'.encode(), b'x'*70000, f'\r\n--{boundary}--\r\n'.encode()]
        response = limited.post('/api/predict',content=iter(chunks),headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
        assert response.status_code == 413


def test_pixel_limit():
    with pytest.raises(ValueError,match='pixel limit'):
        prepare_image(image_bytes(size=(100,100)),max_pixels=9999,input_shape=[1,224,224,3])


def test_missing_models_are_explicit(tmp_path):
    with TestClient(create_app(Settings(model_dir=tmp_path))) as unavailable:
        assert unavailable.get('/api/health').json()['status'] == 'degraded'
        assert all(not m['available'] for m in unavailable.get('/api/models').json())
        response = unavailable.post('/api/predict',data={'model':'simple_cnn'},files={'file':('leaf.png',image_bytes())})
        assert response.status_code == 503
        assert str(tmp_path) not in response.text


def test_cors(client):
    allowed = client.options('/api/predict',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST'})
    assert allowed.headers['access-control-allow-origin'] == 'http://localhost:5173'
    denied = client.options('/api/predict',headers={'Origin':'https://untrusted.example','Access-Control-Request-Method':'POST'})
    assert 'access-control-allow-origin' not in denied.headers


def test_inference_failure_is_sanitized(client,service,monkeypatch):
    def broken(*args,**kwargs):
        raise RuntimeError('sensitive local detail')
    monkeypatch.setattr(service,'predict',broken)
    response = client.post('/api/predict',data={'model':'simple_cnn'},files={'file':('leaf.png',image_bytes())})
    assert response.status_code == 500
    assert 'sensitive local detail' not in response.text
