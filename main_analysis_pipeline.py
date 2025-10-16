# main_analysis_pipeline.py

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# The user's prompt suggests installing umap-learn, so I'll import UMAP from umap
from umap import UMAP
from ripser import ripser
from persim import plot_diagrams, wasserstein
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_PATHS = {
    'control': 'control_data.csv',
    'overexpression': 'overexpression_data.csv',
    'haploinsufficient': 'haploinsufficient_data.csv'
}
N_COMPONENTS_PCA = 20
N_COMPONENTS_UMAP = 3
SMOOTHING_SIGMA = 2.0

# --- Step 1: Data Loading and Preprocessing ---
def load_and_preprocess(filepath):
    """Loads and preprocesses the spike count data."""
    print(f"Loading and preprocessing {filepath}...")
    # Assuming CSV has neurons as rows, time as columns
    spike_counts = pd.read_csv(filepath, header=None).values

    # Smooth the data across time (axis=1) for each neuron
    smoothed_counts = gaussian_filter1d(spike_counts, sigma=SMOOTHING_SIGMA, axis=1)

    # Standardize (Z-score) each neuron's activity across time
    scaler = StandardScaler()
    # Scaler works on features in columns, so we transpose, scale, and transpose back
    scaled_counts = scaler.fit_transform(smoothed_counts.T).T

    print(f"Data shape: {scaled_counts.shape}")
    return scaled_counts

# --- Step 2: Manifold Learning and ID Estimation ---
def run_manifold_analysis(data_matrix, n_pca, n_umap):
    """Performs PCA and UMAP embedding."""
    print("Running manifold analysis...")
    # Data needs to be in (n_samples, n_features) format for sklearn
    # Here, n_samples = time bins, n_features = neurons
    X = data_matrix.T

    # PCA
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA explained variance with {n_pca} components: {explained_variance[-1]:.3f}")

    # UMAP (can be run on raw data or PCA-reduced data)
    # Running on PCA-reduced data is faster and often better
    umap_reducer = UMAP(n_components=n_umap, n_neighbors=30, min_dist=0.1)
    X_umap = umap_reducer.fit_transform(X_pca)

    return {'pca': X_pca, 'umap': X_umap, 'pca_model': pca}


# --- Main Execution Block ---
if __name__ == '__main__':
    # Load and process data for all groups
    processed_data = {name: load_and_preprocess(path) for name, path in DATA_PATHS.items()}

    # Run manifold analysis for all groups
    # This dictionary will store the results (PCA/UMAP embeddings)
    manifold_results = {name: run_manifold_analysis(data, N_COMPONENTS_PCA, N_COMPONENTS_UMAP)
                        for name, data in processed_data.items()}

    # --- Step 3: Geometric Comparison ---
    print("\n--- Geometric Comparison ---")
    # Procrustes Analysis on PCA embeddings
    # Compare each condition to the control
    control_pca = manifold_results['control']['pca']

    # Ensure point clouds have the same number of points for comparison
    min_time_bins = min(control_pca.shape[0],
                        manifold_results['overexpression']['pca'].shape[0],
                        manifold_results['haploinsufficient']['pca'].shape[0])

    control_pca = control_pca[:min_time_bins, :]
    overexp_pca = manifold_results['overexpression']['pca'][:min_time_bins, :]
    haploin_pca = manifold_results['haploinsufficient']['pca'][:min_time_bins, :]

    mtx1, mtx2_overexp, disparity_overexp = procrustes(control_pca, overexp_pca)
    mtx1, mtx2_haploin, disparity_haploin = procrustes(control_pca, haploin_pca)

    print(f"Procrustes disparity (Control vs Overexpression): {disparity_overexp:.4f}")
    print(f"Procrustes disparity (Control vs Haploinsufficient): {disparity_haploin:.4f}")

    # --- Step 4: Topological Comparison (TDA) ---
    print("\n--- Topological Comparison (TDA) ---")
    # Use the UMAP embedding as it often captures non-linear structure better
    # Subsample data for speed if necessary
    control_umap = manifold_results['control']['umap']
    overexp_umap = manifold_results['overexpression']['umap']
    haploin_umap = manifold_results['haploinsufficient']['umap']

    # Subsample the data to avoid memory issues
    n_points_tda = 500  # Number of points to use for TDA

    control_umap_sub = control_umap[np.random.choice(control_umap.shape[0], n_points_tda, replace=False)]
    overexp_umap_sub = overexp_umap[np.random.choice(overexp_umap.shape[0], n_points_tda, replace=False)]
    haploin_umap_sub = haploin_umap[np.random.choice(haploin_umap.shape[0], n_points_tda, replace=False)]

    # ripser computes persistent homology
    # maxdim=2 computes H0, H1, and H2
    print("Running Ripser for Control...")
    dgms_control = ripser(control_umap_sub, maxdim=2)['dgms']
    print("Running Ripser for Overexpression...")
    dgms_overexp = ripser(overexp_umap_sub, maxdim=2)['dgms']
    print("Running Ripser for Haploinsufficient...")
    dgms_haploin = ripser(haploin_umap_sub, maxdim=2)['dgms']

    # Compare persistence diagrams using Wasserstein distance
    # We are particularly interested in H1 (loops)
    d_h1_overexp = wasserstein(dgms_control[1], dgms_overexp[1], matching=False)
    d_h1_haploin = wasserstein(dgms_control[1], dgms_haploin[1], matching=False)

    print(f"Wasserstein distance for H1 (Control vs Overexpression): {d_h1_overexp:.4f}")
    print(f"Wasserstein distance for H1 (Control vs Haploinsufficient): {d_h1_haploin:.4f}")

    # --- Visualization ---
    # Plot persistence diagrams
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plot_diagrams(dgms_control, show=False)
    plt.title('Control')
    plt.subplot(1, 3, 2)
    plot_diagrams(dgms_overexp, show=False)
    plt.title('Overexpression')
    plt.subplot(1, 3, 3)
    plot_diagrams(dgms_haploin, show=False)
    plt.title('Haploinsufficient')
    plt.suptitle("Persistence Diagrams for H0, H1, H2")
    plt.savefig('persistence_diagrams.png')
    plt.close()

    # Plot 3D manifold embeddings
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(control_umap[:, 0], control_umap[:, 1], control_umap[:, 2], s=1)
    ax1.set_title('Control Manifold')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(overexp_umap[:, 0], overexp_umap[:, 1], overexp_umap[:, 2], s=1)
    ax2.set_title('Overexpression Manifold')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(haploin_umap[:, 0], haploin_umap[:, 1], haploin_umap[:, 2], s=1)
    ax3.set_title('Haploinsufficient Manifold')
    plt.savefig('manifold_embeddings.png')
    plt.close()