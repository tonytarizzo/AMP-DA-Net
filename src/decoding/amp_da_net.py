
import math
import torch
import torch.nn as nn


class AMPNet1DEnhanced(nn.Module):
    def __init__(self, URA_codebook, num_layers=10, num_filters=32, kernel_size=3,
                 use_poisson=True,
                 update_alpha=True,
                 learn_sigma2=False,
                 finetune_K=True,
                 finetune_pi=True,
                 finetune_sigma2=True,
                 K_max=None,
                 blend_init=0.85,
                 per_sample_sigma=False,
                 per_sample_alpha=False):
        super().__init__()
        C = URA_codebook if torch.is_tensor(URA_codebook) else torch.tensor(URA_codebook, dtype=torch.float32)
        assert C.dim() == 2, "URA_codebook must be (n, d)"
        self.register_buffer("C_nt", C)    # (n, d)
        self.n, self.d = C.shape
        self.T = num_layers

        # Options
        self.use_poisson      = use_poisson
        self.update_alpha     = update_alpha
        self.learn_sigma2     = learn_sigma2
        self.finetune_K       = finetune_K
        self.finetune_pi      = finetune_pi
        self.finetune_sigma2  = finetune_sigma2
        self.fixed_Kmax       = K_max
        self.per_sample_sigma = per_sample_sigma
        self.per_sample_alpha = per_sample_alpha

        # Per-layer small calibrations (near-identity)
        self.res_scale_raw = nn.Parameter(torch.zeros(self.T))   # residual correction multiplier (~1)
        self.inv_scale_raw = nn.Parameter(torch.zeros(self.T))   # inverse-variance multiplier (~1)
        self.damp_raw      = nn.Parameter(torch.full((self.T,), torch.logit(torch.tensor(0.3))))  # init damp≈0.3
        self.mix_raw       = nn.Parameter(torch.full((self.T,), math.log(blend_init/(1-blend_init))))  # CNN blend
        self.log_tau       = nn.Parameter(torch.full((self.T,), torch.log(torch.tensor(1.25))))   # posterior temp

        # Per-layer alpha mixing gate: sigmoid(alpha_mix_raw[t]) ∈ (0,1)
        self.alpha_mix_raw = nn.Parameter(torch.full((self.T,), torch.logit(torch.tensor(0.2))))

        # Learned per-layer step sizes for parameter fine-tuning
        # sK ∈ (0,2), sPi ∈ (0,1), sSig ∈ (0,1)
        self.k_step_raw     = nn.Parameter(torch.zeros(self.T))
        self.pi_step_raw    = nn.Parameter(torch.zeros(self.T))
        self.sigma_step_raw = nn.Parameter(torch.zeros(self.T))

        # Small 1D CNN denoisers; channels: [R, sqrt(Vi), m1, sqrt(var), alpha, lambda]
        pad = kernel_size // 2
        Cin = 6
        self.denoisers = nn.ModuleList()
        for _ in range(self.T):
            self.denoisers.append(nn.Sequential(
                nn.Conv1d(Cin, num_filters, kernel_size, padding=pad, bias=False),
                nn.ReLU(inplace=False),
                nn.Conv1d(num_filters, num_filters, kernel_size, padding=pad, bias=False),
                nn.ReLU(inplace=False),
                nn.Conv1d(num_filters, 1, kernel_size, padding=pad, bias=False),
                nn.ReLU(inplace=False)
            ))

        # Caches
        self._cached = {
            "C_dn": None,   # (d, n)
            "C2":   None,   # (d, n) = C_dn ⊙ C_dn
            "C2T":  None,   # (n, d)
            "gram": None,   # (n, n)
            "g_pi": None,   # (n,)
        }
        self._round_params = {
            "K_float": None,     # scalar
            "K_int":   None,     # int used only for logging, never in math
            "sigma2":  None,     # scalar per round
        }
        self._current_codebook_id = None
        self.init_policy   = "const_sigma2"   # or "snr_db"
        self.sigma2_const  = 100.0            # used when init_policy == "const_sigma2"
        self.snr_init_db   = -100.0           # used when init_policy == "snr_db"

    def set_init_const(self, sigma2: float):
        self.init_policy  = "const_sigma2"
        self.sigma2_const = float(sigma2)
        return self

    def set_init_from_snr_db(self, snr_db: float):
        self.init_policy = "snr_db"
        self.snr_init_db = float(snr_db)
        return self

    @staticmethod
    def _scale_centered(param, low, high):
        mid  = 0.5 * (low + high)
        half = 0.5 * (high - low)
        return mid + half * torch.tanh(param)

    @staticmethod
    def _eps():
        return 1e-12

    @staticmethod
    def _normalize_pi(pi):
        s = pi.sum()
        if s <= 0:
            return torch.full_like(pi, 1.0/pi.numel())
        return (pi / s).clamp_min(1e-12)

    @torch.no_grad()
    def update_codebook_cache(self, C_nt, pi=None):
        """Compute C^T, C^2, Gram, and (optionally) the direction Gπ for K MF."""
        C_dn = C_nt.t().contiguous()     # (d, n)
        C2   = C_dn.pow(2)               # (d, n)
        C2T  = C2.t().contiguous()       # (n, d)
        gram = C_nt @ C_nt.t()           # (n, n)
        g_pi = None
        if pi is not None:
            pi_n = self._normalize_pi(pi.flatten())
            g_pi = gram @ pi_n
        self._cached.update(dict(C_dn=C_dn, C2=C2, C2T=C2T, gram=gram, g_pi=g_pi))
        self._current_codebook_id = int(C_nt.data_ptr())
        
    # ---------- per-round initialization ----------
    @torch.no_grad()
    def start_new_round(self, z_round, pi_round, *, external_URA_codebook=None):
        C_nt = external_URA_codebook if external_URA_codebook is not None else self.C_nt
        self.update_codebook_cache(C_nt, pi_round)

        Ka0 = 10.0
        self._round_params["K_float"] = Ka0
        self._round_params["K_int"] = int(round(Ka0))
        if self.init_policy == "const_sigma2":
            sigma2 = float(self.sigma2_const)
        elif self.init_policy == "snr_db":
            snr_lin = 10.0 ** (self.snr_init_db / 10.0)
            if self.per_sample_sigma:
                Py = z_round.pow(2).mean(dim=1, keepdim=True)     # (B,1)
                sigma2 = (Py / (1.0 + snr_lin)).clamp_min(self._eps())
            else:
                Py = z_round.pow(2).mean()
                sigma2 = max(self._eps(), float(Py.item() / (1.0 + snr_lin)))
        else:
            raise ValueError(f"Unknown init_policy: {self.init_policy}")
        self._round_params["sigma2"] = sigma2

    # ---------- discrete Bayes table ----------
    def _discrete_mmse_counts(self, R, Vi, alpha, lam, K_max, tau):
        """
        Exact posterior moments under spike+counts prior with temperature τ.
        R, Vi: (B,n), alpha:(n,), lam:(n,), tau: scalar tensor
        Returns: m1 (B,n), var (B,n), alpha_post (B,n), and 'post' if you want debug
        """
        B, n = R.shape
        m_vals = torch.arange(0, K_max + 1, device=R.device, dtype=R.dtype)
        R_exp  = R.unsqueeze(-1)                               # (B,n,1)
        Vi_exp = Vi.unsqueeze(-1).clamp_min(self._eps())       # (B,n,1)
        m_exp  = m_vals.view(1, 1, -1)                         # (1,1,M+1)

        # Gaussian likelihood (log)
        ll = - (R_exp - m_exp)**2 / (2.0 * Vi_exp)

        # Prior weights (log)
        alphaB = alpha if alpha.dim() == 2 else alpha.unsqueeze(0)  # (B,n) or (1,n)
        logw = torch.empty_like(ll)
        logw[:, :, 0] = torch.log1p(-alphaB)                   # m=0

        if self.use_poisson:
            lam_exp = lam.view(1, n, 1).clamp_min(self._eps())
            log_pois = m_exp * torch.log(lam_exp) - lam_exp - torch.lgamma(m_exp + 1.0)
            logw[:, :, 1:] = torch.log(alphaB).unsqueeze(-1) + log_pois[:, :, 1:]
        else:
            M = max(int(m_vals[-1].item()), 1)
            logw[:, :, 1:] = torch.log(alphaB).unsqueeze(-1) - math.log(M)

        # Posterior table with temperature τ
        log_post = ll + logw
        post = torch.softmax(log_post / tau, dim=2)            # (B,n,M+1)

        # Moments
        m1  = torch.sum(post * m_exp, dim=2)                   # (B,n)
        m2  = torch.sum(post * (m_exp**2), dim=2)              # (B,n)
        var = (m2 - m1**2).clamp_min(self._eps())

        # Posterior activity
        alpha_post = post[:, :, 1:].sum(dim=2).clamp(1e-10, 1-1e-10)
        return m1, var, alpha_post, post

    def forward(self, y, external_URA_codebook=None, pi_round=None, snr_db=None):
        """
        y: (B,d)
        If K_round / snr_db are None, use start_new_round() cached values.
        pi_round: (n,) required (the denoiser needs π to form λ=Kπ).
        """
        device = y.device
        C_nt = external_URA_codebook.to(device) if external_URA_codebook is not None else self.C_nt
        if (self._cached["C_dn"] is None) or (int(C_nt.data_ptr()) != self._current_codebook_id):
            # keep cache coherent
            self.update_codebook_cache(C_nt, pi_round)

        C_dn, C2, C2T = self._cached["C_dn"], self._cached["C2"], self._cached["C2T"]
        B, d = y.shape
        assert d == self.d, f"y is (B,{d}), codebook has d={self.d}"

        # --- Priors / inits ---
        assert pi_round is not None, "pi_round must be provided to forward()"
        pi = self._normalize_pi(pi_round.to(device).flatten())

        Ka = torch.as_tensor(self._round_params["K_float"], device=y.device, dtype=y.dtype)
        K_trace = [Ka.detach().item()]  # logging only
        K_max = (int(self.fixed_Kmax) if self.fixed_Kmax is not None else int(max(8, int((2 * Ka).item()))))

        # λ, α0
        lam = (Ka * pi).clamp_min(self._eps())
        if self.per_sample_alpha:
            alpha0 = (1.0 - torch.exp(-lam)).clamp(1e-6, 1-1e-6).unsqueeze(0).expand(B, -1)  # (B,n)
        else:
            alpha0 = (1.0 - torch.exp(-lam)).clamp(1e-6, 1-1e-6)  # (n,)
        alpha  = alpha0.clone()

        # Messages/state
        x_hat   = torch.zeros(B, self.n, device=device, dtype=y.dtype)
        var_hat = torch.ones(B, self.n, device=device, dtype=y.dtype)
        Z       = y.clone()
        V       = torch.ones(B, d, device=device, dtype=y.dtype)

        # Noise variance init (prefer per-round cache)
        if snr_db is None and self._round_params["sigma2"] is not None:
            sigma2 = float(self._round_params["sigma2"])  # scalar
        else:
            snr_lin = 10.0 ** (float(snr_db) / 10.0)
            if self.per_sample_sigma:
                Py = y.pow(2).mean(dim=1, keepdim=True)               # (B,1)
                sigma2 = (Py / (1.0 + snr_lin)).clamp_min(1e-12)      # (B,1)
            else:
                Py = y.pow(2).mean()
                sigma2 = max(1e-12, float(Py.item() / (1.0 + snr_lin)))
        sigma_trace = [float(torch.as_tensor(sigma2).mean().item())] if torch.is_tensor(sigma2) else [float(sigma2)]

        # ---------- unrolled layers ----------
        for t in range(self.T):
            # Learnable calibrations
            scale_out = self._scale_centered(self.res_scale_raw[t], 0.3, 2.0)
            inv_scale = self._scale_centered(self.inv_scale_raw[t], 0.5, 2.0)
            damp_t    = torch.sigmoid(self.damp_raw[t])                 # (0,1)
            mix_t     = torch.sigmoid(self.mix_raw[t])                  # (0,1)
            tau_t     = torch.exp(self.log_tau[t]).clamp_min(1e-3)      # τ>0

            # === Output (GAMP) ===
            V_new = torch.matmul(var_hat, C2T)                          # (B,d)
            Z_tmp = torch.matmul(x_hat, C_nt)                           # (B,d)
            denom = (sigma2 + V).clamp_min(1e-12)
            corr  = ((y - Z) / denom) * V_new
            Z_new = Z_tmp - scale_out * corr

            # Damping (learnable)
            Z = damp_t * Z + (1 - damp_t) * Z_new
            V = damp_t * V + (1 - damp_t) * V_new

            # === Pseudo-channel (GAMP) ===
            inv  = inv_scale * (1.0 / denom).clamp_min(1e-12)           # (B,d)
            var1 = torch.matmul(inv, C2).clamp_min(1e-12)               # (B,n)
            Vi   = 1.0 / var1                                           # (B,n)
            tmp  = ((y - Z) * inv)
            var2 = torch.matmul(tmp, C_dn)                               # (B,n)
            R    = var2 / var1 + x_hat                                   # (B,n)

            # === Discrete Bayes ===
            m1, v_tab, alpha_post, post = self._discrete_mmse_counts(R, Vi, alpha, lam, K_max, tau=tau_t)

            # Per-layer α mix (prior ↔ posterior)
            if self.update_alpha:
                mix_alpha_t = torch.sigmoid(self.alpha_mix_raw[t])       # (0,1)
                if self.per_sample_alpha:
                    alpha = (1 - mix_alpha_t) * alpha0 + mix_alpha_t * alpha_post  # per-sample α: use the per-sample posterior α_post (B,n)
                else:
                    alpha_batch = alpha_post.mean(dim=0)                 # (n,)
                    alpha = (1 - mix_alpha_t) * alpha0 + mix_alpha_t * alpha_batch  # global α: average over batch to get (n,)
            
            # CNN refine & blend
            sqrt_Vi  = torch.sqrt(Vi.clamp_min(1e-12))
            sqrt_var = torch.sqrt(v_tab.clamp_min(1e-12))
            alphaB   = alpha if alpha.dim() == 2 else alpha.unsqueeze(0).expand(B, -1)
            lamB     = lam.unsqueeze(0).expand(B, -1)

            den_in = torch.stack([R, sqrt_Vi, m1, sqrt_var, alphaB, lamB], dim=1)  # (B,6,n)
            x_cnn  = self.denoisers[t](den_in).squeeze(1)                          # (B,n), ≥0
            x_hat  = (1.0 - mix_t) * m1 + mix_t * x_cnn
            var_hat = v_tab  # keep Bayesian variance for the V recursion

            # ---------- Optional parameter fine-tuning (data-driven, no hand damping) ----------
            # K and π from the current posterior means
            if self.finetune_K or self.finetune_pi:
                K_post  = m1.sum(dim=1).mean()                   # scalar
                pi_post = m1.sum(dim=0)
                pi_post = pi_post / (pi_post.sum().clamp_min(1e-12))

            # K update (learned step in (0,2))
            if self.finetune_K:
                sK = 1.0 if t == 0 else (2.0 * torch.sigmoid(self.k_step_raw[t]))
                Ka = torch.clamp(Ka + sK * (K_post - Ka), 0.0, float(self.n))  # keep Ka as tensor

            # π update (learned step in (0,1))
            if self.finetune_pi:
                sPi = torch.sigmoid(self.pi_step_raw[t])
                pi  = (1.0 - sPi) * pi + sPi * pi_post
                pi  = self._normalize_pi(pi)

            # update logs & K output
            K_trace.append(Ka.detach().item())

            # refresh λ, α0 after K/π moves
            lam = (Ka * pi).clamp_min(self._eps())
            alpha0 = (1.0 - torch.exp(-lam)).clamp(1e-6, 1-1e-6)

            # σ² EM head (overwrite or log-gated blend; per-sample if enabled)
            if self.learn_sigma2:
                denom2   = (1.0 + V / (sigma2 + 1e-12))
                s_term1  = ((y - Z)**2) / (denom2**2)
                s_term2  = sigma2 * V / (V + sigma2 + 1e-12)
                if self.per_sample_sigma:
                    sigma2_em = (s_term1 + s_term2).mean(dim=1, keepdim=True)  # (B,1)
                else:
                    sigma2_em = (s_term1 + s_term2).mean()                      # scalar tensor
                if self.finetune_sigma2:
                    sSig = torch.sigmoid(self.sigma_step_raw[t])                # (0,1) log-gated blend (scale-stable)
                    log_sigma2    = torch.log((sigma2 if torch.is_tensor(sigma2) else torch.tensor(sigma2, device=y.device)).clamp_min(1e-12))
                    log_sigma2_em = torch.log(sigma2_em.clamp_min(1e-12))
                    log_sigma2    = (1.0 - sSig) * log_sigma2 + sSig * log_sigma2_em
                    sigma2        = torch.exp(log_sigma2)
                else:
                    sigma2 = sigma2_em  # hard overwrite with EM
            sigma_trace.append(float(sigma2.mean().item()) if torch.is_tensor(sigma2) else float(sigma2))

            # NaN/Inf guard
            if (torch.isnan(x_hat).any() or torch.isinf(x_hat).any()
                or torch.isnan(Z).any() or torch.isnan(V).any()):
                raise RuntimeError("NaN/Inf in AMPNet1DEnhanced: check scaling/prior.")

        return x_hat, Ka, K_trace, sigma_trace
