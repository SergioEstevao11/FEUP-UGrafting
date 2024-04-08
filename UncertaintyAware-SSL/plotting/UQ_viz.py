import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import seaborn as sns
import pandas as pd
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Overlaying uncertainty on an image
def plot_uncertainty_overlay(image, uncertainty_map):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.imshow(uncertainty_map, cmap='jet', alpha=0.5, interpolation='none')
    plt.title('Uncertainty Overlay')
    plt.colorbar()
    plt.show()

# T-SNE for 2D visualization of data landscape
def visualize_with_tsne(features, uncertainties):
    tsne = TSNE(n_components=2, perplexity=25, random_state=42)  # UMAP can also be used

    # Handle 3D tensors by averaging across the views dimension if necessary
    features = features.cpu() if features.is_cuda else features
    uncertainties = uncertainties.cpu() if uncertainties.is_cuda else uncertainties

    # Average across the 'views' dimension to get a single representation per sample
    features_avg = features.mean(dim=1)  # Shape becomes (64, 128)
    features_std_avg = uncertainties.mean(dim=1)  # Shape becomes (64, 128)

    # Now, features_avg is ready for t-SNE transformation
    # However, for uncertainties, you may want to further process it. Here's an example of averaging the uncertainties:
    uncertainties_avg = features_std_avg.mean(dim=-1)  # Averaging across feature dimension, shape becomes (64,)

    # Detach and convert to NumPy for t-SNE
    features_np = features_avg.detach().numpy()
    uncertainties_np = uncertainties_avg.detach().numpy()
    

    # Fit-transform with t-SNE
    tsne_results = tsne.fit_transform(features_np)
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'TSNE-2d-one': tsne_results[:, 0],
        'TSNE-2d-two': tsne_results[:, 1],
        'Uncertainty': uncertainties_np
    })

    # Plot t-SNE scatter plot colored by uncertainties
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="TSNE-2d-one", y="TSNE-2d-two",
        hue="Uncertainty",
        palette=sns.color_palette("viridis", as_cmap=True),
        data=df,
        legend="full",
        alpha=0.8
    )
    plt.title('t-SNE Visualization of Features Colored by Uncertainty')
    plt.savefig(f"tsne_uncertainty_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.show()

def visualize_with_tsne_3d(features, uncertainties):
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)  # Adjust perplexity as needed

    features = features.cpu() if features.is_cuda else features
    uncertainties = uncertainties.cpu() if uncertainties.is_cuda else uncertainties

    # Average across the 'views' dimension to get a single representation per sample
    features_avg = features.mean(dim=1)  # Shape becomes (64, 128)
    features_std_avg = uncertainties.mean(dim=1)  # Shape becomes (64, 128)

    # Now, features_avg is ready for t-SNE transformation
    # However, for uncertainties, you may want to further process it. Here's an example of averaging the uncertainties:
    uncertainties_avg = features_std_avg.mean(dim=-1)  # Averaging across feature dimension, shape becomes (64,)

    # Detach and convert to NumPy for t-SNE
    features_np = features_avg.detach().numpy()
    uncertainties_np = uncertainties_avg.detach().numpy()    
    # Fit-transform with t-SNE
    tsne_results = tsne.fit_transform(features_np)

    # Since we are dealing with 3D visualization, create a DataFrame for the 3 components
    df = pd.DataFrame({
        'TSNE-3d-one': tsne_results[:, 0],
        'TSNE-3d-two': tsne_results[:, 1],
        'TSNE-3d-three': tsne_results[:, 2],
        'Uncertainty': uncertainties_np
    })

    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Generating a color map based on uncertainty
    norm = plt.Normalize(df['Uncertainty'].min(), df['Uncertainty'].max())
    colors = plt.cm.viridis(norm(df['Uncertainty']))

    sc = ax.scatter(df['TSNE-3d-one'], df['TSNE-3d-two'], df['TSNE-3d-three'], c=colors, marker='o')

    # Manually create an axes for colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size as needed
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Uncertainty')

    ax.set_xlabel('TSNE-3d-one')
    ax.set_ylabel('TSNE-3d-two')
    ax.set_zlabel('TSNE-3d-three')
    plt.title('3D t-SNE Visualization of Features Colored by Uncertainty')
    plt.subplots_adjust(right=0.9)  # Adjust the main figure to make space for colorbar
    plt.savefig(f"3d_tsne_uncertainty_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.show()



def visualize_with_3d_histogram(features, uncertainties):
    # Assuming features is a Tensor and uncertainties is a Tensor of standard deviations
    features = features.cpu().numpy() if features.is_cuda else features.numpy()
    uncertainties = uncertainties.cpu().numpy() if uncertainties.is_cuda else uncertainties.numpy()

    # Flatten the uncertainties if they are not already 1D
    if uncertainties.ndim > 1:
        uncertainties = np.mean(uncertainties, axis=1)

    # Apply t-SNE to reduce dimensions to three
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Create a figure for the histogram
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Compute the histogram data
    hist, edges = np.histogramdd(tsne_results, bins=20, density=True, weights=uncertainties)

    # Generate bin centers from edges
    xedges, yedges, zedges = edges
    x_pos = 0.5 * (xedges[:-1] + xedges[1:])
    y_pos = 0.5 * (yedges[:-1] + yedges[1:])
    z_pos = 0.5 * (zedges[:-1] + zedges[1:])

    xpos, ypos, zpos = np.meshgrid(x_pos, y_pos, z_pos, indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = zpos.flatten()
    dx = dy = dz = np.ones_like(xpos) * (xedges[1] - xedges[0])

    # Normalize uncertainties for color mapping
    colors = plt.cm.viridis((hist - hist.min()) / (hist.max() - hist.min()))

    # Plotting the histogram
    ax.bar3d(xpos, ypos, np.zeros_like(zpos), dx, dy, zpos, color=colors.flatten(), zsort='average')
    plt.title('3D Histogram with t-SNE and Uncertainty Coloring')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')

    # Adding a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=hist.min(), vmax=hist.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.5, aspect=20)
    cbar.set_label('Uncertainty')

    plt.show()

# Example usage:
# Assuming `features` is a tensor of shape (N, 3) and `features_std` is a tensor of shape (N, 3)
# where N is the number of samples and each sample has 3 features.
# visualize_with_3d_histogram(features, features_std)
