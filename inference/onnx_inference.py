# Importing required packages
import onnxruntime
import torch
import torch.onnx
from segmentation_models_pytorch import Unet


class ONNXInference:

    def __init__(self, pytorch_model=None, pytorch_weight_file_path=None,
                 model_dummy_input=(1, 3, 480, 640), onnx_weight_path=None):
        self.pytorch_model = pytorch_model
        self.pytorch_weight_file_path = pytorch_weight_file_path
        self.model_dummy_input = model_dummy_input
        self.onnx_weight_path = onnx_weight_path

    def onnxruntime_infer(self):
        # Checking if CUDA is available else use the CPU
        providers = onnxruntime.get_available_providers()
        cuda_available = 'CUDAExecutionProvider' in providers

        if cuda_available:  # If CUDA is available
            # ONNX Model Inference
            ort_session = onnxruntime.InferenceSession(self.onnx_weight_path, providers=['CUDAExecutionProvider'])

            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            onnxruntime_input = {k.name: to_numpy(self.model_dummy_input) for k in ort_session.get_inputs()}
            onnxruntime_output = ort_session.run(None, onnxruntime_input)

        else:  # If CUDA is not available, run on CPU
            # ONNX Model Inference
            ort_session = onnxruntime.InferenceSession(self.onnx_weight_path, providers=['CPUExecutionProvider'])

            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            onnxruntime_input = {k.name: to_numpy(self.model_dummy_input) for k in ort_session.get_inputs()[0]}
            onnxruntime_output = ort_session.run(None, onnxruntime_input)
        return onnxruntime_output

    def pytorch_infer_comparison(self, onnx_output):

        # Loading PyTorch Model and Weight File
        model = self.pytorch_model
        # Load the state dictionary from the saved PyTorch Weight
        state_dict = torch.load(self.pytorch_weight_file_path)
        model.load_state_dict(state_dict)  # Load the state dictionary into the PyTorch model

        # Output from the Model
        torch_output = model(self.model_dummy_input)

        # Checking if the output of PyTorch Model is similar to ONNX Model
        assert torch_output.shape == onnx_output[
            0].shape, "Shape mismatch between PyTorch and ONNX Runtime outputs"

        torch.testing.assert_close(
            torch_output,
            torch.tensor(onnx_output[0]),
            rtol=1e-03,
            atol=1e-05
        )

        print("PyTorch and ONNX Runtime output matched!")
        print(f"Output shape: {torch_output.shape}")
        print(f"Sample output: {onnx_output[0]}")


if __name__ == '__main__':
    onnx_infer = ONNXInference(pytorch_model=Unet(encoder_name="resnet34", encoder_weights="imagenet",
                                                  in_channels=3, classes=5),
                               pytorch_weight_file_path='../pytorch_weights/Aug21_bestmodel_run1_0.038.pt',
                               model_dummy_input=(1, 3, 1824, 2752),
                               onnx_weight_path='../onnx_weights/Aug21_bestmodel_run1_0.038.onnx')

    output = onnx_infer.onnxruntime_infer()
    onnx_infer.pytorch_infer_comparison(output)
