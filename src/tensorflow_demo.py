"""Minimal TensorFlow model used for discovery testing."""

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy")

# Run one inference so TensorFlow is actually imported and used.
sample = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
prediction = model(sample)

print("TensorFlow:", tf.__version__)
print("Prediction:", prediction.numpy())
