import autorootcwd
import src.archs
import torch
from src.utils.registry import ARCH_REGISTRY

# Retrieve the SwinUNETR model from the registry
SwinUNETR = ARCH_REGISTRY.get("SwinUNETR")(
    img_size=(96, 96, 96),  # Example input size
    in_channels=4,          # Example input channels
    out_channels=3,         # Example output channels
    feature_size=24         # Example feature size
)

UNETR = ARCH_REGISTRY.get("UNETR")(
    img_size=(96, 96, 96),  # Example input size
    in_channels=4,          # Example input channels
    out_channels=3,         # Example output channels
    feature_size=24         # Example feature size
)

VNET = ARCH_REGISTRY.get("VNet")(
    in_channels=4,
    out_channels=3
)

UNETPLUSPLUS = ARCH_REGISTRY.get("UNetPlusPlus")(
    in_channels=4,
    out_channels=3
)

# What is happening
print("==============================================")
print("Unit Testing Neural Network Architectures")
print("Author: Kanghyun Ryu (khryu@kist.re.kr)")
print("==============================================")
print("This script is intended for unit testing neural network architectures defined in src.archs.")
print("It is designed for the BRATS21 dataset, which has four channels and a 96x96x96 volume.")
print("The following models will be tested:")
print("1. SwinUNETR")
print("2. UNETR")
print("3. VNet")
print("4. UNetPlusPlus")
print("==============================================")

# Generate dummy input data
dummy_input = torch.randn(1, 4, 96, 96, 96)  # Batch size of 1, 4 channels, 96x96x96 volume

# Define a dummy target for loss computation
dummy_target = torch.randn(1, 3, 96, 96, 96)  # Batch size of 1, 3 channels, 96x96x96 volume

# Define a loss function
criterion = torch.nn.MSELoss()

# Function to test forward and backward pass
def test_model(model, model_name):
    # Forward pass
    output = model(dummy_input)
    

    print(f"====== Testing {model_name} ======")
    print(f"{model_name} Output shape: {output.shape}")
    
    # Compute loss
    loss = criterion(output, dummy_target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients of the first layer
    first_layer_grad = list(model.parameters())[0].grad
    if torch.isnan(first_layer_grad).any():
        print(f"{model_name} First layer gradient contains NaN values.")
    else:
        print(f"{model_name} Test successful: No NaN values in the gradient.")

# Run tests for each model
test_model(SwinUNETR, "SwinUNETR")
test_model(UNETR, "UNETR")
test_model(VNET, "VNET")
test_model(UNETPLUSPLUS, "UNETPlusPlus")