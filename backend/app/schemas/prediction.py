from typing import Literal
from pydantic import BaseModel, Field

ModelId = Literal['simple_cnn', 'mobilenetv3']


class ClassScore(BaseModel):
    class_name: str
    score: float = Field(ge=0, le=1)


class ImageInfo(BaseModel):
    width: int
    height: int
    format: str
    size_bytes: int


class Timings(BaseModel):
    preprocessing_ms: float
    inference_ms: float
    total_ms: float


class PredictionResponse(BaseModel):
    request_id: str
    created_at: str
    model: ModelId
    model_name: str
    predicted_class: str
    prediction_score: float = Field(ge=0, le=1)
    top_3_predictions: list[ClassScore]
    all_predictions: list[ClassScore]
    image: ImageInfo
    input_shape: list[int]
    timings: Timings
    model_sha256: str
    note: str


class ModelInfo(BaseModel):
    id: ModelId
    name: str
    description: str
    available: bool
    status_message: str | None = None
    class_names: list[str]
    input_shape: list[int]
    preprocessing: dict
    parameter_count: int | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metrics: dict[str, float | int] | None = None
    split_fingerprint: str | None = None
