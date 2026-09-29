# AI/ML Discovery Demo

A small repository for testing AI/ML inventory discovery.

It contains:
- Simple TensorFlow and PyTorch code.
- Model/artifact files with extensions commonly used by AI/ML frameworks.
- A generator script that can create real model artifacts when the required packages are installed.
- Tiny placeholder files for extensions that are difficult to generate without large dependencies.

## Run the code

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/tensorflow_demo.py
python src/pytorch_demo.py
python src/create_artifacts.py
```

## Discovery targets

| Extension | Example location | Framework / usage |
|---|---|---|
| `.pb` | `artifacts/tensorflow/model.pb` | TensorFlow |
| `.h5` | `artifacts/tensorflow/model.h5` | Keras |
| `.tflite` | `artifacts/tensorflow/model.tflite` | TensorFlow Lite |
| `.ckpt` | `artifacts/tensorflow/model.ckpt` | TensorFlow checkpoint |
| `.saved_model` | `artifacts/tensorflow/model.saved_model` | TensorFlow SavedModel-style artifact |
| `.pt` | `artifacts/pytorch/model.pt` | PyTorch |
| `.pth` | `artifacts/pytorch/model.pth` | PyTorch |
| `.onnx` | `artifacts/other/model.onnx` | ONNX |
| `.pkl` | `artifacts/other/model.pkl` | scikit-learn |
| `.joblib` | `artifacts/other/model.joblib` | scikit-learn |
| `.model` | `artifacts/other/model.model` | XGBoost / LightGBM / CatBoost-style |
| `.bin` | `artifacts/other/model.bin` | LightGBM / CatBoost-style |
| `pytorch_model.bin` | `artifacts/huggingface/` | Hugging Face Transformers |
| `tf_model.h5` | `artifacts/huggingface/` | Hugging Face Transformers |
| `config.json` | `artifacts/huggingface/` | Hugging Face Transformers |
| `tokenizer.json` | `artifacts/huggingface/` | Hugging Face Transformers |
| `vocab.txt` | `artifacts/huggingface/` | Hugging Face Transformers |

## Important

The files already committed under `artifacts/` are intentionally tiny discovery fixtures.
Some are placeholders because a genuine model artifact can be much larger.

`src/create_artifacts.py` creates genuine small artifacts where practical.
This is useful if your discovery mechanism validates file contents instead of only
checking filenames/extensions.

The repository is intended for a controlled test environment.
