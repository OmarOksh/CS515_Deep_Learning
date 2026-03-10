import os
import torch
from torchviz import make_dot
from models.MLP import MLP

def generate_model_viz():
    """
    Generates a computational graph visualization of the MLP model using torchviz.
    """
    # Ensure the plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Instantiate your baseline MLP model
    # Input size: 784 (28x28 for MNIST)
    # Hidden sizes: [512, 256]
    # Classes: 10
    model = MLP(
        input_size=784, 
        hidden_sizes=[512, 256], 
        num_classes=10, 
        dropout=0.3, 
        activation="relu"
    )
    # Switch to evaluation mode so BatchNorm layers use running stats
    model.eval()
    
    # Create a dummy input tensor matching the shape of a single MNIST image
    # (Batch size of 1, 1 channel, 28x28 pixels)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Pass the dummy input through the model to build the computational graph
    output = model(dummy_input)
    
    # Generate the visualization graph
    # show_attrs=True displays tensor sizes, show_saved=True shows saved tensors
    dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    
    # Save the graph as a PNG in the plots directory
    dot.format = 'png'
    save_path = "plots/mlp_architecture"
    dot.render(save_path)
    
    print(f"Model architecture visualization saved to {save_path}.png")

if __name__ == "__main__":
    generate_model_viz()