"""Create small, real AI/ML artifacts for discovery testing.

Run:
    python src/create_artifacts.py

The script intentionally creates tiny artifacts so the repository stays small.
"""

from pathlib import Path
import json
import pickle

ROOT = Path(__file__).resolve().parents[1]
TF_DIR = ROOT / "artifacts" / "tensorflow"
PT_DIR = ROOT / "artifacts" / "pytorch"
OTHER_DIR = ROOT / "artifacts" / "other"
HF_DIR = ROOT / "artifacts" / "huggingface"

for d in (TF_DIR, PT_DIR, OTHER_DIR, HF_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- TensorFlow ----------------
try:
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    # HDF5/Keras artifact
    model.save(TF_DIR / "real_model.h5")

    # TensorFlow SavedModel (TF 2.x)
    try:
        model.export(TF_DIR / "real_saved_model")
    except AttributeError:
        tf.saved_model.save(model, TF_DIR / "real_saved_model")

    # TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    (TF_DIR / "real_model.tflite").write_bytes(tflite_model)

    # Frozen GraphDef protobuf (.pb)
    concrete = tf.function(model).get_concrete_function(
        tf.TensorSpec([None, 4], tf.float32)
    )
    frozen = tf.graph_util.convert_variables_to_constants_v2(concrete)
    (TF_DIR / "real_model.pb").write_bytes(
        frozen.graph.as_graph_def().SerializeToString()
    )

    # TensorFlow checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.save(str(TF_DIR / "real_model.ckpt"))

    print("Created TensorFlow artifacts.")

except Exception as exc:
    print("TensorFlow artifacts skipped:", exc)

# ---------------- PyTorch ----------------
try:
    import torch
    import torch.nn as nn

    pt_model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )

    # Full model object
    torch.save(pt_model, PT_DIR / "real_model.pt")

    # State dictionary
    torch.save(pt_model.state_dict(), PT_DIR / "real_model.pth")

    # Checkpoint-style file
    torch.save(
        {"model_state_dict": pt_model.state_dict(), "framework": "pytorch"},
        PT_DIR / "real_model.ckpt",
    )

    print("Created PyTorch artifacts.")

except Exception as exc:
    print("PyTorch artifacts skipped:", exc)

# ---------------- Generic / discovery fixtures ----------------
# These are deliberately tiny. They are intended to exercise extension detection.
fixtures = {
    OTHER_DIR / "model.onnx": b"AI_DISCOVERY_FIXTURE_ONNX",
    OTHER_DIR / "model.pkl": pickle.dumps({"framework": "scikit-learn", "demo": True}),
    OTHER_DIR / "model.joblib": b"AI_DISCOVERY_FIXTURE_JOBLIB",
    OTHER_DIR / "model.model": b"AI_DISCOVERY_FIXTURE_MODEL",
    OTHER_DIR / "model.bin": b"AI_DISCOVERY_FIXTURE_BIN",
    HF_DIR / "pytorch_model.bin": b"AI_DISCOVERY_FIXTURE_PYTORCH_BIN",
    HF_DIR / "tf_model.h5": b"AI_DISCOVERY_FIXTURE_TF_H5",
    HF_DIR / "tokenizer.json": json.dumps({"demo": True, "type": "tokenizer"}).encode(),
    HF_DIR / "vocab.txt": b"[PAD]\n[UNK]\nhello\nworld\n",
}

for path, data in fixtures.items():
    path.write_bytes(data)

(HF_DIR / "config.json").write_text(
    json.dumps({
        "model_type": "demo",
        "architectures": ["DemoModel"],
        "hidden_size": 8
    }, indent=2)
)

print("Created generic discovery fixtures.")
