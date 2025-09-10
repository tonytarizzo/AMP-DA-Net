import numpy as np
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numpy.linalg import svd, norm, eigvals
import psutil


def check_power_constraint(codebook, tol=1e-6):
    norms = np.linalg.norm(codebook, axis=0)
    if np.all(np.abs(norms - 1.0) < tol):
        valid_message = "True: Power constraint is satisfied"
    else:
        max_norm = np.max(norms)
        min_norm = np.min(norms)
        avg_norm = np.mean(norms)
        std_norm = np.std(norms)
        num_violations = np.sum(np.abs(norms - 1.0) >= tol)
        valid_message = (
            "False: Power constraint not satisfied.\n"
            f"Number of violating codewords: {num_violations}/{codebook.shape[1]}\n"
            f"Max norm: {max_norm:.6f}\n"
            f"Min norm: {min_norm:.6f}\n"
            f"Average norm: {avg_norm:.6f}\n"
            f"Norm standard deviation: {std_norm:.6f}"
        )
    return valid_message


def log_resource_usage(stage=None):
    if stage is None:
        stage = ""
    # CPU
    vm = psutil.virtual_memory()
    print(f"[{stage}] CPU: {psutil.cpu_percent()}% util, "
          f"{vm.percent}% RAM used ({vm.used/1e9:.2f} GB / {vm.total/1e9:.2f} GB)")

    # GPU (if you ever run on CUDA)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated()
        resv  = torch.cuda.memory_reserved()
        print(f"[{stage}] GPU: allocated {alloc/1e9:.2f} GB, reserved {resv/1e9:.2f} GB")


def error_feedback_info(deltas, error_matrix):
    delta_l2 = deltas.norm(p=2).item()
    err_l2   = error_matrix.norm(p=2).item()
    avg_delta_abs = deltas.abs().mean().item()
    avg_err_abs   = error_matrix.abs().mean().item()
    ratio = err_l2 / (delta_l2 + 1e-12)
    print(f"Delta Norm = {delta_l2:.3e}, Error Norm = {err_l2:.3e}, Error / Delta = {ratio:.3f}")
    print(f"Average Delta = {avg_delta_abs:.3e}, Average Error = {avg_err_abs:.3e}")


def compute_bits_per_dimension(num_codewords: int, block_size: int) -> float:
    """
    Compute bits per dimension = (log2 M) / Vb
    where M = num_codewords, Vb = block_size.
    """
    return math.log2(num_codewords) / block_size


def compute_ul_overhead(D, quant_len, ura_len, P=1024, complex=True):
    """ Compute the uplink overhead in terms of quantization and resource allocation, assumes packing in real setup. """
    splits = np.ceil(D/quant_len)
    if complex:
        symbols_per_split = ura_len
    else:
        symbols_per_split = np.ceil(ura_len/2)
    total_symbols = splits * symbols_per_split
    return np.ceil(total_symbols / P)


def show_SNR_db(K_a, dim, noise_variance):
    """
    Compute the Signal-to-Noise Ratio (SNR) in dB.

    Parameters:
    - K_a (int): Number of active users (i.e., the nonzero elements in x).
    - dim (int): Dimensionality of the codebook (i.e., the length of received signal y).
    - noise_variance (float): Variance of Gaussian noise added during transmission.

    Returns:
    - float: SNR in decibels (dB).
    """
    signal_power = K_a / dim
    noise_power = noise_variance
    return 10 * np.log10(signal_power / noise_power)


def transmission_SNR_db(clean_sig, noise_vec):
    """
    Calculate SNR per sample, then return tensor of SNR values in dB.
    
    Args:
        clean_sig: (batch_size, signal_dim) - clean signal
        noise_vec: (batch_size, signal_dim) - noise vector
    
    Returns:
        torch.Tensor: (batch_size,) - SNR in dB for each sample
    """
    # Calculate power per sample (sum over signal dimension)
    sig_power = (clean_sig ** 2).sum(dim=1)      # (batch_size,)
    noise_power = (noise_vec ** 2).sum(dim=1)    # (batch_size,)
    
    # Clamp to avoid division by zero or log of zero
    noise_power = torch.clamp(noise_power, min=1e-12)
    
    # Calculate SNR per sample
    snr_ratio = sig_power / noise_power           # (batch_size,)
    snr_db = 10.0 * torch.log10(snr_ratio)       # (batch_size,)
    return snr_db

def snr_db(clean_sig, *, fading_variance=None, noise_variance=None):
    p_signal = clean_sig.pow(2).mean(dim=1)
    if fading_variance is None or fading_variance == 0:
        snr_lin = p_signal / noise_variance
    else:
        if clean_sig.is_complex():
            second_moment = 2.0 * fading_variance    # E[|h|²] for complex
        else:
            second_moment = fading_variance          # E[h²] for real
        snr_lin = second_moment * p_signal / noise_variance
    return 10.0 * torch.log10(snr_lin.clamp(min=1e-12))

def setting_info(n, dim, noise_variance, fading_variance, dataset):
    device = dataset.device
    with torch.no_grad():
        # 1) fixed random codebook, unit-norm columns (dim × n)
        C_fixed = F.normalize(torch.randn(dim, n, device=device), p=2, dim=0)
        z = dataset @ C_fixed.T                                                

        # 2) SNR per sample (dB)
        if fading_variance is not None and fading_variance > 0.0:
            snr_batch = snr_db(z, fading_variance=fading_variance, noise_variance=noise_variance)
        else:
            snr_batch = snr_db(z, noise_variance=noise_variance)                       
        print(f"Dataset SNR (dB): {snr_batch.mean().item():.2f} ± {snr_batch.std().item():.2f}")

        # 3) Eb/N0 per sample
        E_s = (z**2).sum(dim=1)                                                  
        N0 = 2.0 * noise_variance
        k_bits = math.log2(n)                                                     
        EbN0 = torch.clamp(E_s / (k_bits * N0), min=1e-12)                        
        EbN0_db = 10.0 * torch.log10(EbN0)                                           
        print(f"Dataset Eb/N0 (dB): {EbN0_db.mean().item():.2f} ± {EbN0_db.std().item():.2f}")

        # 4) Maximum message length
        max_bits = math.floor(k_bits)
        print(f"Max message length: {max_bits} bits (log2(n) ≈ {k_bits:.2f})")
    return 


def visualise_x_distribution(generator_func, n, K_a, num_trials=1000):
    all_values = []
    
    for _ in range(num_trials):
        x = generator_func(n, K_a)
        all_values.extend(x[x > 0])  # Collect all nonzero values
    
    # Count frequency of each value
    unique_values, counts = np.unique(all_values, return_counts=True)
    
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, counts, width=0.8, color="skyblue", edgecolor="black")
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of Values in Generated Sparse Vectors ({num_trials} Trials)", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


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

### Older methods for comparison, kept for reference 
# ----------------------------------------------
def show_nonzero_indices(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return np.sort(np.where(x != 0)[0])
    elif x.ndim == 2:
        return [np.sort(np.where(row != 0)[0]) for row in x]
    else:
        raise ValueError(f"Unsupported input dimension {x.ndim}, expected 1 or 2")


def show_nonzero_values(x):
    return x[x != 0]


def compare_similarity_once(x, x_recovered): # Old method for comparison
    x = np.array(x)
    x_recovered = np.array(x_recovered)

    matches = np.intersect1d(x, x_recovered, assume_unique=False)
    return len(matches) / x.size


def show_active_elements_diff(x, x_recovered):
    return len(show_nonzero_values(x)) - len(show_nonzero_values(x_recovered))


def show_SNR_db_old(y, noise_variance): 
    signal_power = np.linalg.norm(y)**2 / y.size
    noise_power = noise_variance
    return 10 * np.log10(signal_power / noise_power)


def evaluate_encoding_decoding(
    generate_x_func,
    generate_codebook_func,
    encode_func,
    add_noise_func,
    decode_func,
    n,
    dim,
    K_a,
    noise_variance,
    num_iterations=100,
    similarity_func=None,
    sparsity_diff_func=None,
    compute_nmse=None,
    compute_accuracy=None,
    compute_rsnr=None,
):
    """
    Evaluates the accuracy of the encoding and decoding process over multiple iterations.

    Args:
        generate_x_func: Function to generate sparse input vectors.
        generate_codebook_func: Function to generate the random codebook.
        encode_func: Function to encode the sparse input vector using the codebook.
        add_noise_func: Function to add transmission noise to the encoded signal.
        decode_func: Function to decode the noisy received signal.
        n: Dimensionality of the sparse vector.
        dim: Dimensionality of the codebook.
        K_a: Total sum constraint for the sparse vector.
        noise_variance: Variance of the added Gaussian noise.
        num_iterations: Number of iterations to run the evaluation.
        similarity_func: (Optional) Function to compute similarity between true and recovered indices.
        sparsity_diff_func: (Optional) Function to compute the difference in sparsity.

    Returns:
        A dictionary with the results:
            - Average similarity score.
            - Average sparsity difference.
    """
    similarity_scores = []
    sparsity_differences = []
    nmses = []
    accuracies = []
    rsnrs = []

    for _ in tqdm(range(num_iterations), desc="Running iterations"):
        # Generate data
        x = generate_x_func(n, K_a)
        codebook = generate_codebook_func(n, dim)
        encoded_signal = encode_func(x, codebook)
        noisy_signal = add_noise_func(encoded_signal, noise_variance)
        x_recovered = decode_func(noisy_signal, codebook, K_a)

        # Compute similarity and sparsity difference if functions are provided
        if similarity_func:
            similarity_scores.append(similarity_func(show_nonzero_indices(x), show_nonzero_indices(x_recovered)))
        if sparsity_diff_func:
            sparsity_differences.append(sparsity_diff_func(x, x_recovered))
        if compute_nmse:
            nmse = compute_nmse(x, x_recovered)
            nmses.append(nmse)
        if compute_accuracy:
            accuracy = compute_accuracy(x, x_recovered)
            accuracies.append(accuracy)
        if compute_rsnr:
            rsnr = compute_rsnr(x, x_recovered)
            rsnrs.append(rsnr)

    # Calculate averages
    avg_similarity = np.mean(similarity_scores) if similarity_scores else None
    avg_sparsity_diff = np.mean(sparsity_differences) if sparsity_differences else None
    avg_nmse = np.mean(nmses) if nmses else None
    avg_accuracy = np.mean(accuracies) if accuracies else None
    avg_rsnr = np.mean(rsnrs) if rsnrs else None

    print("Average Similarity Score:", avg_similarity)
    print("Average Sparsity Difference:", avg_sparsity_diff)
    print("Average NMSE:", avg_nmse)
    print("Average Accuracy:", avg_accuracy)
    print("Average RSNR:", avg_rsnr)

    return {
        "average_similarity_score": avg_similarity,
        "average_sparsity_difference": avg_sparsity_diff,
        "average_nmse": avg_nmse,
        "average_accuracy": avg_accuracy,
        "average_rsnr": avg_rsnr,
    }


def print_recovery_results(x, x_recovered, similarity_func=None):
    """
    Prints detailed recovery results for the original and recovered sparse vectors.

    Args:
        x: Original sparse vector.
        x_recovered: Recovered sparse vector.
        C: (Optional) Codebook matrix, used for power constraint check.
        similarity_func: (Optional) Function to calculate similarity between original and recovered indices.
    """
    print("-------------------")
    print("Active user indices:", show_nonzero_indices(x))
    print("Recovered Active User Indices:", show_nonzero_indices(x_recovered))
    
    print("-------------------")
    print("Active user values:", show_nonzero_values(x))
    print("Recovered Active User Values:", show_nonzero_values(x_recovered))
    
    print("-------------------")
    print("Original Sum: ", x.sum())
    print("Recovered Sum: ", x_recovered.sum())
    # Uncomment the next line if you want rounded sums
    # print("Rounded Recovered Sum: ", np.round(x_recovered).sum())
    
    print("-------------------")
    print("Active User Sparsity:", len(show_nonzero_values(x)))
    print("Recovered Active User Sparsity:", len(show_nonzero_values(x_recovered)))
    
    if similarity_func:
        similarity_score = similarity_func(show_nonzero_indices(x), show_nonzero_indices(x_recovered))
        print("Similarity Score: ", similarity_score)
    
    print("-------------------")

