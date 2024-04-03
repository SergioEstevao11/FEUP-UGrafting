import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import seaborn as sns
import pandas as pd

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
    tsne = TSNE(n_components=2, random_state=42) # UMAP can be used as well
    tsne_results = tsne.fit_transform(features)
    
    df = pd.DataFrame({
        'TSNE-2d-one': tsne_results[:,0],
        'TSNE-2d-two': tsne_results[:,1],
        'Uncertainty': uncertainties
    })
    
    plt.figure(figsize=(10,8))
    sns.scatterplot(
        x="TSNE-2d-one", y="TSNE-2d-two",
        hue="Uncertainty",
        palette=sns.color_palette("viridis", as_cmap=True),
        data=df,
        legend="full",
        alpha=0.8
    )
    plt.title('t-SNE Visualization of Features Colored by Uncertainty')
    plt.show()