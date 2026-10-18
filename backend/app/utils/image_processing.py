"""Exact Pillow preprocessing used by both supplied notebook exports.

Both exports take raw float32 RGB [0,255]. Their DIFFERENT normalization
layers are embedded in the saved models; never normalize a second time here.
"""
import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def prepare_image(data: bytes, *, max_pixels: int, input_shape: list[int]) -> tuple[np.ndarray, dict]:
    if not data:
        raise ImageValidationError('The uploaded file is empty.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                image_format = source.format
                if image_format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise ImageValidationError('Please upload a JPEG, PNG, or WebP image.', 415)
                if source.width * source.height > max_pixels:
                    raise ImageValidationError(f'Image exceeds the {max_pixels:,}-pixel limit.', 413)
                if getattr(source, 'n_frames', 1) != 1:
                    raise ImageValidationError('Animated or multi-frame images are not supported. Upload one still image.')
                source.verify()
            with Image.open(io.BytesIO(data)) as source:
                source.load()
                image = ImageOps.exif_transpose(source)
                width, height = image.size
                if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                    rgba = image.convert('RGBA')
                    image = Image.alpha_composite(Image.new('RGBA', rgba.size, (0, 0, 0, 255)), rgba)
                image = image.convert('RGB')
                target_height, target_width = input_shape[1:3]
                scaled = ImageOps.contain(image, (target_width, target_height), Image.Resampling.BILINEAR)
                canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                canvas.paste(scaled, ((target_width - scaled.width) // 2, (target_height - scaled.height) // 2))
                tensor = np.asarray(canvas, dtype=np.float32)[None]
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError('Image dimensions are too large.', 413) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError('The file is not a valid, complete image. Please choose another image.') from exc
    return tensor, {'width': width, 'height': height, 'format': image_format, 'size_bytes': len(data)}
