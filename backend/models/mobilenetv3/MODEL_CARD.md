# Rice leaf disease classifier

Backbone: ImageNet-pretrained MobileNetV3Large, alpha 1.0, 224x224.
Winning stage: stage_b_fine_tune.
Selected mobile variant: float16.
Cleaned images: 4805.
Train / validation / test: 3355 / 716 / 734.
Exact class order: ["Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf Blast", "Leaf scald", "Sheath Blight"].
Images without known capture-group metadata: 100.0%.

## Evaluation
Keras test accuracy: 0.8937.
Keras test macro F1: 0.8832.
Mobile test accuracy: 0.8937.
Mobile test macro F1: 0.8832.
These are internal held-out results, not independent field performance estimates.
Model, epoch, and mobile format selection used validation data only.

## Intended use and limits
Research prototype for one principal supported rice leaf/sheath condition per photo.
Six-class scores are uncalibrated. High confidence does not establish a diagnosis.
Healthy means no visible symptoms among supported labels, not absence of all disease.
Unknown diseases, mixed infections, non-rice inputs and poor images can be misclassified.
Near-duplicate grouping cannot discover every same-plant, same-farm or transformed image.
Expert label review, independent smartphone field testing and device benchmarking remain necessary.
No pesticide recommendation, clinical-style probability, severity estimate or validated rejection
mechanism is provided. Link disease labels to a separately reviewed local agricultural guide.

## Files
rice_disease.keras: full float32 Keras inference model; use compile=False to load.
rice_disease.tflite: selected mobile export; see mobile_io.json for tensor types.
class_names.json / labels.txt: exact output order.
preprocessing.json / predict.py: image preprocessing and Python prediction reference.
reference_cases: synthetic preprocessing parity case (not an accuracy test).
reports: audit records, split manifest, learning curves, metrics and prediction tables.
training_environment.txt: full environment snapshot, not a minimal deployment dependency list.

## Provenance
Archive SHA256: 1c201cb503cb69c642e32e14ff7d5470bae5c77bbfcfbcd138a440b7df65ff95.
Class labels are derived from user-provided folders and have not been independently confirmed.
Record dataset source/licence and agricultural reviewer in your project documentation.
