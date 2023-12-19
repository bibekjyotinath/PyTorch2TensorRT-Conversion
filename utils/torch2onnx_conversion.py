# Importing Required Packages to convert PyTorch Models to ONNX Model
import torch.onnx
import onnx
from segmentation_models_pytorch import Unet


class Torch2ONNXConversion:

    def __init__(self, model, weight_file_path, model_input, onnx_save_path):
        self.model = model
        self.weight_file_path = weight_file_path
        self.model_input = model_input
        self.onnx_save_path = onnx_save_path

    def torch2onnx_conversion(self):
        print('================================== Converting Pytorch Model to ONNX ==================================')
        # Loading PyTorch Model and Weight File
        model = self.model
        print('Model Loaded Successfully!!')
        model.load_state_dict(torch.load(self.weight_file_path))
        print('Weight File Loaded Successfully!!')
        # Input of the Model (batch, channel, height, width)
        input = torch.randn(self.model_input, requires_grad=True)

        # Output from the Model
        torch_output = model(input)
        assert torch_output.shape[0] == self.model_input[0]
        assert not torch.isnan(torch_output).any(), 'Output included NaNs'

        print('Model Conversion Started ...')
        # ONNX Model Conversion and Saving the ONNX Model
        onnx_save = self.onnx_save_path

        torch.onnx.export(model,  # Model
                          input,  # Model Input
                          onnx_save,  # Save Location of ONNX Weight
                          verbose=True,  # Detailed output or logging during the execution of a function
                          export_params=True,  # Store the trained parameter weights inside the model file
                          opset_version=11,  # ONNX version to export the model to
                          do_constant_folding=False,  # Whether to execute constant folding for optimization
                          input_names=['input'],  # The Model's input names
                          output_names=['torch_output'],  # The Model's output names
                          dynamic_axes={
                              'input': {
                                  0: 'batch_size'
                              },  # Variable Length Axes
                              'torch_output': {
                                  0: 'batch_size'
                              }
                          })

        print('ONNX Model Conversion is Complete ..... ')

        print('Validating ONNX Model ..... ')
        try:
            onnx.checker.check_model(onnx.load(self.onnx_save_path))
            print("ONNX model is valid.")
        except onnx.onnx_cpp2py_export.checker.ValidationError as e:
            print(f"ONNX model is invalid. Error: {e}")

        return input


if __name__ == '__main__':
    t2o = Torch2ONNXConversion(
        model=Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=5),
        weight_file_path='../pytorch_weights/Aug21_bestmodel_run1_0.038.pt',
        model_input=(1, 3, 1824, 2752),
        onnx_save_path='../onnx_weights/Aug21_bestmodel_run1_0.038.onnx')
    torch_input = t2o.torch2onnx_conversion()
