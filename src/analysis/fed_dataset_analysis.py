
import os
import torch
import matplotlib.pyplot as plt
from .metrics import kl_divergence, normalized_entropy, gini_coefficient, topk_mass

def analyze_dataset(path, rounds_to_plot=None, save_dir="analysis_results"):
    data = torch.load(path)
    X, K_rounds, pi_targets, pi_estimates = data['X'], data['K_rounds'], data['pi_targets'], data['pi_estimates']
    os.makedirs(save_dir, exist_ok=True)

    # 1) Round-by-round K_a 
    rounds = list(range(1, len(K_rounds) + 1))
    Ka_vals = K_rounds.tolist()

    plt.figure(figsize=(10,6))
    plt.plot(rounds, Ka_vals, linewidth=2, linestyle='-', color='tab:blue')
    plt.xlabel("Round")
    plt.ylabel("Kₐ (active devices)")
    plt.title("Round-by-round Kₐ")
    plt.grid(True, linestyle='--', alpha=0.5)
    path_ka = os.path.join(save_dir, "round_by_round_Ka.png")
    plt.tight_layout()
    plt.savefig(path_ka, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Round-by-round Kₐ plot saved to {path_ka}")

    # 2) Bias / concentration metrics per round
    n = X.shape[1]
    uniform = torch.full((n,), 1.0 / n, device=X.device)
    metrics = {'round': [], 'K_a': [], 'KL_target': [], 'KL_estimate': [],'normalized_entropy_target': [],'normalized_entropy_estimate': [], 'gini_target': [], 'gini_estimate': [], 'top_5%_mass_target': [], 'top_5%_mass_estimate': []}

    for idx, r in enumerate(rounds):
        p_t, p_e = pi_targets[idx], pi_estimates[idx]
        K_a = int(K_rounds[idx])
        kl_t, kl_e = kl_divergence(p_t, uniform), kl_divergence(p_e, uniform)
        ne_t, ne_e = normalized_entropy(p_t), normalized_entropy(p_e)
        g_t, g_e = gini_coefficient(p_t), gini_coefficient(p_e)
        top5_t, top5_e = topk_mass(p_t, fraction=0.05), topk_mass(p_e, fraction=0.05)
        metrics['round'].append(r)
        metrics['K_a'].append(K_a)
        metrics['KL_target'].append(kl_t)
        metrics['KL_estimate'].append(kl_e)
        metrics['normalized_entropy_target'].append(ne_t)
        metrics['normalized_entropy_estimate'].append(ne_e)
        metrics['gini_target'].append(g_t)
        metrics['gini_estimate'].append(g_e)
        metrics['top_5%_mass_target'].append(top5_t)
        metrics['top_5%_mass_estimate'].append(top5_e)

    # Print summary stats
    print(f"\nDataset Analysis Summary:")
    print(f"Total rounds: {len(rounds)}")
    print(f"Codeword space size (n): {n}")
    print(f"K_a range: {min(Ka_vals)} - {max(Ka_vals)}")

    # Trend plots
    def save_trend(y_t, y_e, ylabel, title, fname):
        plt.figure(figsize=(10,6))
        plt.plot(metrics['round'], y_t, linewidth=2, label='Target', color='tab:blue')
        plt.plot(metrics['round'], y_e, linewidth=2, label='Estimated', color='tab:orange')
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        full_path = os.path.join(save_dir, fname)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{title} plot saved to {full_path}")

    save_trend(metrics['KL_target'], metrics['KL_estimate'], "KL Divergence", "KL Divergence to Uniform over Rounds", "kl_divergence_over_rounds.png")
    save_trend(metrics['normalized_entropy_target'], metrics['normalized_entropy_estimate'], "Entropy / log(n)", "Normalized Entropy over Rounds", "normalized_entropy_over_rounds.png")
    save_trend(metrics['gini_target'], metrics['gini_estimate'], "Gini", "Gini Coefficient over Rounds", "gini_over_rounds.png")
    save_trend(metrics['top_5%_mass_target'], metrics['top_5%_mass_estimate'], "Top 5% Mass", "Top 5% Mass over Rounds", "top5_mass_over_rounds.png")

    # 3) Select rounds to plot if not specified
    if rounds_to_plot is None:
        total_rounds = len(rounds)
        if total_rounds >= 5:
            indices = [0, int(0.25*total_rounds), int(0.5*total_rounds), int(0.75*total_rounds), total_rounds-1]
            rounds_to_plot = [rounds[i] for i in indices]
        else:
            rounds_to_plot = rounds
    print(f"\nRounds selected for detailed plots: {rounds_to_plot}")

    # Overlay codeword distributions for selected rounds
    plt.figure(figsize=(10,6))
    for r in rounds_to_plot:
        p_t = pi_targets[r-1]
        sorted_p, _ = torch.sort(p_t, descending=True)
        plt.plot(sorted_p.cpu().numpy(), label=f"Round {r}", linewidth=2)
    plt.plot([0, n-1], [1.0 / n, 1.0 / n], linestyle='--', label='Uniform (1/n)', color='gray', linewidth=2)
    plt.xlabel("Codeword rank (descending)")
    plt.ylabel("Probability")
    plt.title("Sorted Empirical Codeword Distributions") 
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    dist_path = os.path.join(save_dir, "codeword_distributions.png")
    plt.tight_layout()
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Codeword distribution overlay plot saved to {dist_path}")

    # Pi estimates distribution for selected rounds
    plt.figure(figsize=(10,6))
    for r in rounds_to_plot:
        p_est = pi_estimates[r-1]
        p_est_sorted, _ = torch.sort(p_est, descending=True)
        plt.plot(p_est_sorted.cpu().numpy(), linestyle='--', label=f"Est π Round {r}", linewidth=2)
    plt.xlabel("Codeword rank")
    plt.ylabel("Estimated π")
    plt.title("Sorted Estimated π Distributions per Round")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    est_path = os.path.join(save_dir, "estimated_pi_distributions.png")
    plt.tight_layout()
    plt.savefig(est_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Estimated π plot saved to {est_path}")

    # Lorenz target curve
    plt.figure(figsize=(10,6))
    for r in rounds_to_plot:
        p_t = pi_targets[r-1]
        sorted_p, _ = torch.sort(p_t)  # ascending
        cum = torch.cumsum(sorted_p, dim=0)
        cum = torch.cat([torch.tensor([0.0], device=cum.device), cum])  # start at 0
        x = torch.linspace(0, 1, len(cum))
        plt.plot(x.cpu().numpy(), cum.cpu().numpy(), label=f"Round {r}", linewidth=2)
    plt.plot([0,1],[0,1], 'k--', label='Equality line', linewidth=2)
    plt.xlabel("Fraction of codewords (sorted ascending)")
    plt.ylabel("Cumulative probability")
    plt.title("Lorenz Curves of Target Codeword Bias")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    lorenz_path = os.path.join(save_dir, "lorenz_curves_target.png")
    plt.tight_layout()
    plt.savefig(lorenz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Lorenz target curve plot saved to {lorenz_path}")

    # Lorenz estimated curve
    plt.figure(figsize=(10,6))
    for r in rounds_to_plot:
        p_e = pi_estimates[r-1]
        sorted_p, _ = torch.sort(p_e)  # ascending
        cum = torch.cumsum(sorted_p, dim=0)
        cum = torch.cat([torch.tensor([0.0], device=cum.device), cum])  # start at 0
        x = torch.linspace(0, 1, len(cum))
        plt.plot(x.cpu().numpy(), cum.cpu().numpy(), label=f"Round {r}", linewidth=2)
    plt.plot([0,1],[0,1], 'k--', label='Equality line', linewidth=2)
    plt.xlabel("Fraction of codewords (sorted ascending)")
    plt.ylabel("Cumulative probability")
    plt.title("Lorenz Curves of Estimated Codeword Bias")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    lorenz_path = os.path.join(save_dir, "lorenz_curves_estimate.png")
    plt.tight_layout()
    plt.savefig(lorenz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Lorenz estimate curve plot saved to {lorenz_path}")
    return metrics
