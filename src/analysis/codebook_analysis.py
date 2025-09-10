
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import svd, norm, eigvals

def visualise_codebook_heatmap(C, title="Codebook Heatmap", save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(C, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Codebook Columns")
    plt.ylabel("Codebook Rows")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def analyse_codebook(C, plot=True, k_rip=None, save_path=None, show_plots=True):
    """
    Analyse a codebook matrix for wireless transmission with easy-to-understand metrics.
    
    Parameters:
    -----------
    C : ndarray
        Codebook matrix where columns are codewords
    plot : bool, optional
        Whether to generate visualization plots (default: True)
    k_rip : int, optional
        Sparsity parameter for RIP estimation (default: min(10, C.shape[1]//4))
        
    Returns:
    --------
    dict
        Dictionary of metrics with clear descriptions
    """
    if isinstance(C, torch.Tensor):
        C = C.detach().cpu().numpy()
        
    m, n = C.shape  # m: signal dimension, n: number of codewords
    metrics = {}
    
    # Normalize columns to unit norm
    C_normalized = C / (norm(C, axis=0, keepdims=True) + 1e-8)
    
    # 1. Basic properties
    metrics['dimensions'] = f"{m}x{n} (signal dimension x codebook size)"
    metrics['dynamic_range'] = f"{np.max(np.abs(C)) / (np.min(np.abs(C[C != 0])) + 1e-10):.2f}"
    
    # 2. Cross-correlation analysis
    G = np.abs(np.dot(C_normalized.T, C_normalized))
    np.fill_diagonal(G, 0)  # Zero out diagonal for cross-correlation analysis
    
    # Key cross-correlation statistics
    metrics['mutual_coherence'] = f"{np.max(G):.4f}"
    metrics['avg_cross_correlation'] = f"{np.mean(G):.4f}"
    metrics['median_cross_correlation'] = f"{np.median(G):.4f}"
    
    # Cross-correlation percentiles for distribution understanding
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        metrics[f'cross_corr_p{p}'] = f"{np.percentile(G.flatten(), p):.4f}"
    
    # 3. SVD analysis for condition number and stability
    U, S, Vt = svd(C, full_matrices=False)
    metrics['condition_number'] = f"{S[0] / (S[-1] + 1e-10):.4f}"
    metrics['rank_effective'] = np.sum(S > 1e-10)
    metrics['energy_top_50pct'] = np.sum(S[:len(S)//2]) / np.sum(S)
    
    # 4. RIP analysis (Restricted Isometry Property)
    if k_rip is None:
        k_rip = min(10, n // 4)  # Default sparsity parameter for RIP

    def compute_num_trials(n, dim, k_rip, confidence_level=0.99, error_margin=0.05):
        """
        n: number of codewords, dim: dimensionality, k_rip: sparsity
        confidence_level: desired confidence (e.g., 0.99 for 99%)
        error_margin: acceptable error in RIP constant estimation
        """
        from scipy.special import comb
        import math
        
        # Base number based on Hoeffding's inequality for confidence bounds
        failure_prob = 1 - confidence_level
        base_trials = int((math.log(2/failure_prob) / (2 * error_margin**2)))
        complexity_factor = math.sqrt(k_rip * math.log(n/k_rip))
        num_trials = int(base_trials * complexity_factor)
        print(f"Number of trials used to estimate RIP constant: {num_trials}")
        return num_trials
    
    # RIP estimation through random subspace sampling
    rip_deltas = []
    num_trials = compute_num_trials(n, m, k_rip)
    
    for _ in range(num_trials):
        # Randomly select k_rip columns
        indices = np.random.choice(n, k_rip, replace=False)
        C_sub = C[:, indices]
        
        # Compute restricted isometry constant for this submatrix
        gram_sub = C_sub.T @ C_sub
        eigs = eigvals(gram_sub)
        delta = max(abs(np.max(eigs) - 1), abs(1 - np.min(eigs)))
        rip_deltas.append(delta)
    
    metrics['rip_delta_median'] = f"{np.median(rip_deltas):.4f} (for k={k_rip})"
    metrics['rip_delta_max'] = f"{np.max(rip_deltas):.4f} (for k={k_rip})"
    metrics['rip_quality'] = "Good" if np.median(rip_deltas) < 0.5 else "Fair" if np.median(rip_deltas) < 0.7 else "Poor"
    
    # 5. Orthogonality assessment
    I = np.eye(n)
    gram = np.dot(C_normalized.T, C_normalized)
    metrics['orthogonality_error'] = f"{norm(gram - I) / norm(I):.4f}"
    
    # Visual output if requested
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Cross-correlation heatmap
        sns.heatmap(G, ax=axes[0,0], cmap='viridis', vmin=0, vmax=min(1, np.max(G)*1.5))
        axes[0,0].set_title("Cross-correlation Magnitudes")
        
        # Plot 2: Cross-correlation distribution
        axes[0,1].hist(G.flatten(), bins=50)
        axes[0,1].axvline(np.median(G), color='r', linestyle='--', label=f"Median: {np.median(G):.4f}")
        axes[0,1].set_title("Cross-correlation Distribution")
        axes[0,1].set_xlabel("Correlation Value")
        axes[0,1].legend()
        
        # Plot 3: Singular value spectrum
        axes[1,0].semilogy(range(1, len(S)+1), S, '-o')
        axes[1,0].set_title("Singular Value Spectrum")
        axes[1,0].set_xlabel("Index")
        axes[1,0].set_ylabel("Singular Value (log scale)")
        
        # Plot 4: RIP estimates
        axes[1,1].hist(rip_deltas, bins=20)
        axes[1,1].axvline(np.median(rip_deltas), color='r', linestyle='--', 
                       label=f"Median δ: {np.median(rip_deltas):.4f}")
        axes[1,1].set_title(f"RIP Constant Distribution (k={k_rip})")
        axes[1,1].set_xlabel("δ Value")
        axes[1,1].legend()
        
        plt.tight_layout()
        metrics['plots'] = fig
        if save_path:
            plt.savefig(save_path)
        if show_plots:
            plt.show()
    
    # Summary section with most important metrics
    metrics['summary'] = {
        'mutual_coherence': float(metrics['mutual_coherence']),
        'condition_number': float(metrics['condition_number']),
        'rip_delta_median': float(metrics['rip_delta_median'].split()[0]),
        'orthogonality_error': float(metrics['orthogonality_error']),
        'avg_cross_correlation': float(metrics['avg_cross_correlation'])
    }
    
    return metrics

def print_codebook_analysis(metrics, print_full=False):
    """
    Print a concise summary of codebook analysis metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from analyze_codebook function
    print_full : bool
        Whether to print all metrics or just the summary
    """
    print("=" * 50)
    print("CODEBOOK ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Dimensions: {metrics['dimensions']}")
    print(f"Mutual coherence: {metrics['mutual_coherence']} (lower is better, ideal: 0)")
    print(f"Average cross-correlation: {metrics['avg_cross_correlation']} (lower is better)")
    print(f"RIP quality: {metrics['rip_quality']} (δ = {metrics['rip_delta_median'].split()[0]})")
    print(f"Condition number: {metrics['condition_number']} (lower is better)")
    print(f"Orthogonality error: {metrics['orthogonality_error']} (lower is better)")
    
    if print_full:
        print("\n" + "=" * 50)
        print("DETAILED METRICS")
        print("=" * 50)
        
        for key, value in metrics.items():
            if key not in ['plots', 'summary'] and not isinstance(value, np.ndarray):
                print(f"{key}: {value}")