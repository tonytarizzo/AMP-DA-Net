
from .metrics import (
    kl_divergence,
    entropy,
    normalized_entropy,
    gini_coefficient,
    topk_mass,
    _code_sim_abs_cosine,
    _bandedness_score,
    compute_bits_per_dimension,
    compute_ul_overhead,
    error_feedback_info,
)
from .parameter_estimation import (
    compute_pi_distribution,
    pi_from_indices,
    estimate_K_mf,
    m2_from_indices,
    m2_from_indices_counts,
    m2_from_idxmini_u_stat,
    m2_true_from_pi,
)
from .codebook_analysis import (
    visualise_codebook_heatmap,
    analyse_codebook,
    print_codebook_analysis,
)
from .fed_dataset_analysis import (
    analyze_dataset,
)
from .logging import (
    _to_py,
    save_json,
    append_jsonl,
    save_round_history_csv,
    run_manifest,
    _triplet_to_dict,
)
from .visualisations import (
    plot_codebook_ordering,
    plot_dataset_collection_history,
    plot_pretraining,
    plot_estimation_diagnostics,
    plot_parameter_traces,
    plot_inference_history,
)