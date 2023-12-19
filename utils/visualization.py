import onnx
import time
import netron
import torch
from torchviz import make_dot
from torchsummary import summary
from segmentation_models_pytorch import Unet


class ModelVisualizer:

    def __init__(self, onnx_weight=None, pytorch_model=None, pytorch_weight=None, dummy_input=None):
        self.onnx_weight_path = onnx_weight
        self.pytorch_weight_path = pytorch_weight
        self.dummy_input = dummy_input
        self.pytorch_model = pytorch_model

    def onnx_network_visualizer(self):
        # Loading ONNX Model and Checking Model Conversion
        onnx_model = self.onnx_weight_path
        onnx.checker.check_model(onnx_model)

        # Netron Visualizer to display architecture of ONNX Model
        netron.start(self.onnx_weight_path, browse=True)
        time.sleep(1)  # Sleeping for 1 sec after closing the browser
        netron.stop()  # Stopping the Netron visualizer

        print('Model is Valid and Adheres to the ONNX Specification ..... ')

    def pytorch_network_visualizer(self):
        # Load PyTorch Model and weights
        self.pytorch_model.load_state_dict(torch.load(self.pytorch_weight_path))

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pytorch_model.to(device)
        self.pytorch_model.eval()

        # Display model summary
        summary(self.pytorch_model, input_size=self.dummy_input[1:],
                device=str(device))  # Exclude batch size from input_size

        # Create a dummy input tensor (batch size of 1)
        dummy_input = torch.randn(1, *self.dummy_input[1:]).to(device)

        # Visualize the model using torchviz
        dot = make_dot(self.pytorch_model(dummy_input), params=dict(self.pytorch_model.named_parameters()))
        dot.render(filename='model_visualization', format='png', cleanup=True)


if __name__ == '__main__':
    mv = ModelVisualizer(onnx_weight='/home/scorpion/PycharmProjects/pytorch2tensorrt/onnx_weights'
                                     '/Aug21_bestmodel_run1_0.038.onnx',
                         pytorch_model=Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3,
                                            classes=5),
                         pytorch_weight='/home/scorpion/PycharmProjects/pytorch2tensorrt/pytorch_weights'
                                        '/Aug21_bestmodel_run1_0.038.pt', dummy_input=(1, 3, 480, 640))
    mv.onnx_network_visualizer()
    mv.pytorch_network_visualizer()
