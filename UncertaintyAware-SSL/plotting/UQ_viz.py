import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import seaborn as sns
import pandas as pd
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle


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

    # (50000, 2, 128) - 50,000 samples, 2 views, 128 features

    # Limit the data for practical visualization purposes
    features = features[:100]
    uncertainties = uncertainties[:100]  # Ensure this matches the size of features being processed

    features = features.cpu() if features.is_cuda else features
    uncertainties = uncertainties.cpu() if uncertainties.is_cuda else uncertainties

    # Average across the 'views' dimension to get a single representation per sample
    features_avg = features.mean(dim=1)  # Shape becomes (64, 128)
    features_std_avg = uncertainties.mean(dim=1)  # Shape becomes (64, 128)

    # Check the dimensionality and adjust if necessary
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)  # Flattening the last two dimensions

    if uncertainties.ndim > 1:
        uncertainties = uncertainties.flatten()  # Ensuring uncertainties is 1D

    print("Features shape:", features.shape)  # Debug: Print shapes
    print("Uncertainties shape:", uncertainties.shape)

    # Apply t-SNE to reduce dimensions to three
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(features)

    print("t-SNE Results shape:", tsne_results.shape)  # Debug: Print t-SNE results shape

    # Compute the histogram data
    hist, edges = np.histogramdd(tsne_results, bins=20, weights=uncertainties, density=True)
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
    norm = plt.Normalize(vmin=hist.min(), vmax=hist.max())
    colors = plt.cm.viridis(norm(hist.flatten()))

    # Plotting the histogram
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, np.zeros_like(zpos), dx, dy, hist.flatten(), color=colors, zsort='average')
    plt.title('3D Histogram with t-SNE and Uncertainty Coloring')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')

    # Adding a color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Uncertainty Density')
    plt.savefig(f"3dhistogram_tsne_uncertainty_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.show()

def visualize_with_tsne_3d_histogram(features, uncertainties, epoch):
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)

    features = features[:100]
    uncertainties = uncertainties[:100]
    
    features = features.cpu() if features.is_cuda else features
    uncertainties = uncertainties.cpu() if uncertainties.is_cuda else uncertainties

    # Average across dimensions
    features_avg = features.mean(dim=1)
    uncertainties_avg = uncertainties.mean(dim=1).mean(dim=-1)

    # Detach and convert to NumPy
    features_np = features_avg.detach().numpy()
    uncertainties_np = uncertainties_avg.detach().numpy()

    # Fit-transform with t-SNE
    tsne_results = tsne.fit_transform(features_np)

    # Compute the histogram data
    hist, edges = np.histogramdd(tsne_results, bins=20, weights=uncertainties_np, density=True)

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
    norm = plt.Normalize(vmin=hist.min(), vmax=hist.max())
    colors = plt.cm.viridis(norm(hist.flatten()))

    # Plotting the histogram
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, np.zeros_like(zpos), dx, dy, hist.flatten(), color=colors, zsort='average')

    # Adding a color bar properly
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)  # Use the 3D Axes for the colorbar
    cbar.set_label('Uncertainty Density')

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.title('3D Histogram of t-SNE Results Colored by Uncertainty')
    plt.subplots_adjust(right=0.9)  # Adjust to make space for colorbar
    plt.savefig(f"./plots/3d_tsne_histogram_{epoch}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

def linegraph_minmax_area(std_data, epochs): #std_data[(min, max, mean, std_points)]

    if torch.is_tensor(std_data):
        std_data = std_data.cpu().detach().numpy()
    elif isinstance(std_data, list) and all(torch.is_tensor(x) for x in std_data):
        # If std_data is a list of tensors, convert each to CPU numpy, and then concatenate
        std_data = np.concatenate([x.cpu().detach().numpy() for x in std_data], axis=0)
    
    if torch.is_tensor(epochs):
        epochs = epochs.cpu().detach().numpy()
    # Create the plot
    epochs = [i for i in range(1, epochs + 1)]

    average_values = [item[2] for item in std_data]
    min_values = [item[0] for item in std_data]
    max_values = [item[1] for item in std_data]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, average_values, label='Average Metric', color='b', linewidth=2)  # Line plot for average values
    #plt.fill_between(epochs, min_values, max_values, color='gray', alpha=0.1, label='Min/Max Range')  # Shaded area for range

    #add the std_points, like in a scatterplot, with a smaller size
    std_points = [item[3] for item in std_data]

    std_points_averaged = [torch.mean(x, dim=(1,2)) for x in std_points]
    # transform the list of tensors into a list of numpy arrays
    std_points_averaged = [x.numpy() for x in std_points_averaged]
    print(std_points_averaged)
    
    for index, std_values in enumerate(std_points_averaged):
        # Create an array filled with the current index for the x-values
        x_values = np.full_like(std_values, index+1)
        
        # Scatter plot for this group of std deviations
        plt.scatter(x_values, std_values, alpha=0.05, color='r')  # Adjust the size as needed


    #plt.scatter(epochs, average_values, s=std_points_averaged, color='r', alpha=0.3, label='Standard Deviation Points')  # Scatter plot for std points

    # Adding labels and title
    plt.title('Uncertainty Progression Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Uncertainty (variance)')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.savefig(f"./linegraph_minmax_area_{epochs[-1]}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

max_uncertainty_bound = 1.0  # Set the maximum bound for uncertainty values

def plot_2d_histogram(encodings, uncertainties, epoch=1):
    # Select the first index of dim=1 for both encodings and uncertainties
    encodings = encodings[:, 0, :].reshape(-1, 128)  # Reshape encodings to (10000, 128)
    uncertainties = uncertainties[:, 0, :].reshape(-1, 128)  # Reshape uncertainties to (10000, 128)
    
    # Use the mean uncertainty per encoding as the weight for the histogram
    uncertainties_mean = np.mean(uncertainties, axis=1)

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    reduced_encodings = pca.fit_transform(encodings)

    # Compute the 2D histogram with bins and weights
    x = reduced_encodings[:, 0]  # First principal component
    y = reduced_encodings[:, 1]  # Second principal component
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100, weights=uncertainties_mean, density=False)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Determine the color scale's max uncertainty bound (first epoch initialization)
    if epoch == 1:
        global max_uncertainty_bound
        max_uncertainty_bound = np.max(uncertainties_mean)

    # Create the plot
    fig, ax = plt.subplots()
    norm = Normalize(vmin=0, vmax=max_uncertainty_bound)  # Fixed bounds for the color scale
    cmap = plt.get_cmap('viridis')
    cbar = ScalarMappable(norm=norm, cmap=cmap)

    cax = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, aspect='auto', interpolation='none')
    ax.set_title(f'2D Histogram of Reduced Encodings Colored by Uncertainty - Epoch {epoch}')
    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')

    # Adding colorbar
    fig.colorbar(cbar, ax=ax, label='Uncertainty Density')

    # Show plot
    plt.savefig(f"./plot_2d_histogram_epoch_{epoch}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.close(fig)  # Close the figure to free memory

 

def plot_precision_recall_curve_multiclass(true_labels, prob_predictions, n_classes):
    # Binarize the labels for multiclass
    true_labels_bin = label_binarize(true_labels, classes=range(n_classes))
    
    # Print shapes for debugging
    print(f"true_labels_bin.shape: {true_labels_bin.shape}")
    print(f"prob_predictions.shape: {prob_predictions.shape}")
    
    # Check if predictions are already in one-hot format (required shape [n_samples, n_classes])
    if prob_predictions.ndim == 1 or prob_predictions.shape[1] == 1:
        raise ValueError("prob_predictions should have shape [n_samples, n_classes] for multiclass classification.")
    
    # Check dimensions match
    if true_labels_bin.shape[1] != prob_predictions.shape[1]:
        raise ValueError("Dimension mismatch between converted labels and predictions.")
    
    # Setup plot details
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'lime'])
    plt.figure(figsize=(7, 7))

    # Compute Precision-Recall and plot each class
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], prob_predictions[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'Precision-Recall curve of class {i} (area = {auc(recall, precision):0.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="upper right")
    plt.show()

def plot_average_precision_recall_curve(true_labels, prob_predictions, n_classes):
    true_labels_bin = label_binarize(true_labels, classes=range(n_classes))
    
    # Micro-averaging (binary classif. approach) -> agregates true positives, false positives, and false negatives across all classes then computes metrics
    precision_micro, recall_micro, _ = precision_recall_curve(true_labels_bin.ravel(), prob_predictions.ravel())
    average_precision_micro = average_precision_score(true_labels_bin, prob_predictions, average='micro')

    # Macro-averaging -> computes each metric individually and then does the average
    precision_macro = []
    recall_macro = np.linspace(0, 1, 100)
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], prob_predictions[:, i])
        precision_interp = np.interp(recall_macro, recall[::-1], precision[::-1])
        precision_macro.append(precision_interp)
        
    precision_macro = np.mean(precision_macro, axis=0)
    average_precision_macro = average_precision_score(true_labels_bin, prob_predictions, average='macro')

    plt.figure(figsize=(12, 8))

    # Micro-averaging plot
    plt.plot(recall_micro, precision_micro, color='b', lw=2, label=f'Micro-average (area = {average_precision_micro:.2f})')

    # Macro-averaging plot
    plt.plot(recall_macro, precision_macro, color='r', lw=2, label=f'Macro-average (area = {average_precision_macro:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()