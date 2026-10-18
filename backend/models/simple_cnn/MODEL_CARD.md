# Rice leaf simple CNN

Architecture: 3x3 convolutions with 32, 64, 128 filters; ReLU and 2x2 max pooling after each;
global average pooling; Dense 128/ReLU; dropout 0.3; Dense 6/Softmax.
110,534 trainable parameters. Random initialization, no pretrained weights.
Selection: lowest validation loss, including L2 penalty 0.0001.
Train / validation / test images: 3355 / 716 / 734.
Class order: ["Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf Blast", "Leaf scald", "Sheath Blight"].
Test accuracy: 0.833787.
Test macro F1: 0.824204.
Old split imported: False.
Mobile export validated: True.
Dataset archive SHA256: 1c201cb503cb69c642e32e14ff7d5470bae5c77bbfcfbcd138a440b7df65ff95.

Use predict.py for the reference EXIF/RGB/aspect-preserving preprocessing.
Input is float32 [0,255]; the model normalizes internally.
This is an internal held-out evaluation, not independent field validation.
One principal supported condition per photograph is assumed. Capture metadata coverage:
0.0%. Near-duplicate grouping cannot discover every related photo.
Scores are uncalibrated and unsupported conditions/non-rice inputs may receive high scores.
Record the dataset source, licence and label review separately.

Files: rice_leaf_simple_cnn.keras, class_names.json, labels.txt, predict.py,
preprocessing.json, inference_requirements.txt, experiment_config.json and reports/.
rice_leaf_simple_cnn.tflite is included only after a successful float32 parity check.
Training epoch backups remain in the Modal run directory and are not in this bundle.
