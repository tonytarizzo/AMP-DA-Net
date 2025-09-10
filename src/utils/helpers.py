
import numpy as np
import torch
import psutil

def split_rounds_concatenated(T: torch.Tensor, round_len: int):
    assert T.shape[0] % round_len == 0, "concatenated length not divisible by round_len"
    R = T.shape[0] // round_len
    return [T[i*round_len:(i+1)*round_len] for i in range(R)]

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

