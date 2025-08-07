import torch
import torch.nn as nn
import numpy as np
import onnx
import tensorflow as tf
from tensorflow import keras
import onnxruntime as ort

# ----- STEP 0: Define the model (basic MLP) -----
class StrokeMLP(nn.Module):
    def __init__(self, input_size):
        super(StrokeMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Define model and dummy input
input_size = 10  # adjust this to match your dataset features
model = StrokeMLP(input_size)
model.eval()

# ----- STEP 1: Export to ONNX -----
onnx_path = "cardio_stroke_model.onnx"
dummy_input = torch.randn(1, input_size)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print(f"Model exported to ONNX at {onnx_path}")

# ----- STEP 2: Load ONNX model -----
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

def _run_onnx_inference(input_data):
    return sess.run([output_name], {input_name: np.ascontiguousarray(input_data)})[0]

# ----- STEP 3: Wrap in TensorFlow model -----
class ONNXModelWrapper(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_size], dtype=tf.float32)])
    def call(self, inputs):
        return tf.py_function(_run_onnx_inference, [inputs], tf.float32)

onnx_wrapper = ONNXModelWrapper()
_ = onnx_wrapper(tf.constant(np.random.randn(1, input_size).astype(np.float32)))

saved_model_path = "tf_saved_model_stroke"
tf.saved_model.save(onnx_wrapper, saved_model_path)
print(f"SavedModel exported at {saved_model_path}")

# ----- STEP 4: Convert to TFLite -----
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.allow_custom_ops = True
tflite_model = converter.convert()

tflite_path = "cardio_stroke_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at {tflite_path}")
