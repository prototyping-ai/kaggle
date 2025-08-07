import torch
import onnx
import onnxruntime as ort
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

# --- Part 1: Export the PyTorch JIT model to ONNX ---

# Load the JIT-compiled model. Note: The path below is a placeholder and should be updated.
device = torch.device("cpu")
try:
    cnn_model = torch.jit.load("custom_cnn_mobile.pt", map_location=device)
    cnn_model.eval()
except FileNotFoundError:
    print("Error: PyTorch model not found. Please verify the path.")
    # Exit or handle the error gracefully
    cnn_model = None

if cnn_model:
    # Create the dummy input tensor with the correct size.
    # The shape should match the expected input of your PyTorch model.
    dummy_input = torch.randn(1, 3, 112, 112, requires_grad=True)

    # Export the JIT model to ONNX format.
    onnx_model_path = "custom_cnn_mobile_model.onnx"
    torch.onnx.export(cnn_model,
                      dummy_input,
                      onnx_model_path,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'}})

    print("Step 1: JIT model has been successfully exported to ONNX.")

    # --- Part 2: Convert ONNX to a Keras model and then to TFLite ---

    # Use ONNX Runtime to get the model's output
    sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    output_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name
    
    # Create a helper function to run ONNX inference, which will be called by tf.py_function
    def _run_onnx_inference(input_data):
        """Helper function to run the ONNX session with a numpy array."""
        # Ensure the input is a contiguous numpy array, which is often required
        # by ONNX Runtime and can resolve data format issues from tf.py_function.
        outputs = sess.run([output_name], {input_name: np.ascontiguousarray(input_data)})
        return outputs[0]

    # Create a simple Keras model that wraps the ONNX Runtime session.
    class ONNXModelWrapper(keras.Model):
        def __init__(self, onnx_session, output_name, input_name):
            super(ONNXModelWrapper, self).__init__()
            self.output_name = output_name
            self.input_name = input_name

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, 3, 112, 112), dtype=tf.float32)
        ])
        def call(self, inputs):
            # Use tf.py_function to wrap the ONNX inference call.
            # We must specify the output shape and dtype to ensure the graph
            # is correctly traced.
            return tf.py_function(
                _run_onnx_inference,
                inp=[inputs],
                Tout=tf.float32,
                # Explicitly set the output shape to allow for proper graph building.
                # The shape can be inferred from the model's output. For this example,
                # we'll assume a known output shape after the first dimension.
                # You might need to adjust this based on your model's output.
                # Example: tf.TensorShape([None, 1000]) for a 1000-class classifier
                # We will return a placeholder for the output shape here
                # since it's difficult to determine without knowing the model.
                # In practice, you would calculate this from the ONNX model's output.
                name="onnx_inference_op"
            )

    # Instantiate the wrapper model.
    onnx_model_wrapper = ONNXModelWrapper(sess, output_name, input_name)
    onnx_model_wrapper.build(input_shape=(None, 3, 112, 112))
    
    # Call the model once with dummy data to build the graph before exporting.
    dummy_input_tf = tf.convert_to_tensor(dummy_input.detach().numpy())
    _ = onnx_model_wrapper(dummy_input_tf)

    # Save the wrapper as a TensorFlow SavedModel.
    tf_saved_model_path = "tf_saved_model_final"
    onnx_model_wrapper.export(tf_saved_model_path)

    print("Step 2: ONNX model successfully wrapped in a Keras model and saved.")

    # --- Part 3: Convert the SavedModel to TFLite, allowing custom ops ---

    tflite_model_path = "custom_cnn_mobile_model.tflite"

    # Convert the SavedModel to a TFLite model using the native converter.
    # The key change here is to set `allow_custom_ops=True`.
    # This instructs the converter to keep `tf.py_function` as a custom operation
    # instead of trying to convert it to a native TFLite op, which caused the error.
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Save the TFLite model to a file.
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print("Step 3: SavedModel successfully converted to TFLite format.")
    print(f"Final TFLite file saved at: {tflite_model_path}")