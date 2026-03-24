import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.MLP import MLP

def get_features(model, dataloader, device):
    """
    Extracts features from the penultimate layer of the MLP.
    """
    model.eval()
    features = []
    labels = []
    
    # Extract all layers except the final Linear classification layer
    # In the baseline MLP, self.net has 10 layers, so we take [:-1]
    feature_extractor = torch.nn.Sequential(*list(model.net.children())[:-1])
    
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            out = feature_extractor(imgs)
            features.append(out.cpu().numpy())
            labels.append(lbls.numpy())
            
            # Limit to ~2000 samples to keep t-SNE processing fast
            if len(features) * dataloader.batch_size > 2000: 
                break
                
    return np.vstack(features), np.concatenate(labels)

def plot_tsne():
    os.makedirs("plots", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Rebuild the baseline model structure
    model = MLP(input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout=0.3, activation="relu")
    
    # Load your best baseline weights
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Please run your baseline training first.")
        return
        
    model.to(device)
    
    # Load a batch of the MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)
    
    print("Extracting features from the penultimate layer...")
    features, labels = get_features(model, test_loader, device)
    
    print("Running t-SNE dimensionality reduction (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("t-SNE Visualization of MLP Hidden Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    save_path = "plots/tsne_visualization.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved t-SNE plot to {save_path}")

if __name__ == "__main__":
    plot_tsne()