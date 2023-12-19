# Importing required packages
import tensorrt as trt
from onnx_inference import ONNXInference

class TensorRTInference:

    def __init__(self, onnx_file=None, engine_file=None, model_dummy_input=(1, 3, 480, 640)):
        self.onnx_file = onnx_file
        self.engine_file = engine_file
        self.model_dummy_input = model_dummy_input

    # def tensorrt_infer(self):

