"""Environment configuration, resolved independently of the working directory."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / '.env')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: (BASE_DIR / os.getenv('MODEL_DIR', 'models')).resolve())
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv('MAX_UPLOAD_MB', '15')) * 1024 * 1024)
    max_image_pixels: int = field(default_factory=lambda: int(os.getenv('MAX_IMAGE_PIXELS', '25000000')))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        origin.strip() for origin in os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if origin.strip()
    ))

    def __post_init__(self):
        if self.max_upload_bytes <= 0 or self.max_image_pixels <= 0:
            raise ValueError('Upload and image limits must be positive.')


settings = Settings()
