
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.onnx
from  model.FlexiNet import FlexiNet

def load_model(model_path):
    # Initialize your model. Ensure all parameters match with how the model was defined during training
    model = FlexiNet(input_channels=1, num_classes=1)
    # Load the model state dict
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Adjust 'cpu' to 'cuda' if using GPU
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Path where the trained model checkpoint is saved
model_path = "../pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth" 
model = load_model(model_path)

# Set the model to evaluation mode
model.eval()

# Generate a dummy input that matches the input shape that the model expects
dummy_input = torch.randn(4, 13, 1, 64, 64)  # Adjust the size according to your model's input

# Path to save the ONNX model
onnx_model_path = 'exported_models/checkpoint_epoch_100_nuimage_L1_best.onnx'

# Export the model
torch.onnx.export(model,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  onnx_model_path,     # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=17,    # the ONNX version to export the model to
                  input_names=['input'],     # the model's input names
                  output_names=['output'],   # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable-length axes
                                'output': {0: 'batch_size'}})

print(f'Model successfully saved to {onnx_model_path}')