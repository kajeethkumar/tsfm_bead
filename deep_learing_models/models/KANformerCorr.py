import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


# =============================================================================
# 1.  KAN Layer (B-spline parameterisation, from kan-ad.py extended to 2-D)
# =============================================================================

class KANLayer(nn.Module):
    """
    A single KAN layer.

    Each connection i→j is modelled by a learnable univariate function
    parameterised as a linear combination of K B-spline (here: Fourier cosine)
    basis functions, exactly as in Eq. (10)-(11) of the paper.

        h^{l+1}_j = Σ_i  φ^{l}_{j,i}( h^{l}_i )
        φ^{l}_{j,i}(x) = Σ_k  c^{l}_{j,i,k} · B_k(x)

    We use the Fourier-cosine basis from KAN-AD because it is smooth, global,
    and numerically stable, while still being learnable per edge.
    """

    def __init__(self, in_features: int, out_features: int, num_basis: int = 10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = num_basis  # number of basis functions

        # Learnable coefficients  c^{l}_{j,i,k}  shape: (out, in, K)
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * 0.01
        )

        # Basis frequency indices  k = 1 … K  (fixed)
        self.register_buffer(
            "freqs",
            torch.arange(1, num_basis + 1, dtype=torch.float32)  # (K,)
        )

        # Residual projection when in_features ≠ out_features
        self.residual = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.GELU()

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate K cosine basis functions for each scalar in x.

        Args:
            x : (..., in_features)
        Returns:
            B : (..., in_features, K)
        """
        # x: (..., in)  →  (..., in, 1)  →  (..., in, K)
        x_exp = x.unsqueeze(-1)                           # (..., in, 1)
        freqs = self.freqs.view(*([1] * x.dim()), -1)     # (1, …, 1, K)
        return torch.cos(freqs * x_exp)                   # (..., in, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, in_features)  OR  (batch, T, in_features)
        Returns:
            out: (batch, out_features) OR (batch, T, out_features)
        """
        shape = x.shape
        if x.dim() == 3:
            B, T, D = shape
            x_flat = x.reshape(B * T, D)          # (B*T, in)
        else:
            x_flat = x                             # (B, in)

        Bx = self.basis(x_flat)                    # (B*T, in, K)
        # φ_{j,i}(x_i) = Σ_k  c_{j,i,k} · B_k(x_i)
        # einsum: (B*T, in, K) × (out, in, K) → (B*T, out, in) → sum over in
        phi = torch.einsum("bik,oik->boi", Bx, self.coeffs)  # (B*T, out, in)
        out = phi.sum(dim=-1)                      # (B*T, out)  Eq. (10)

        # Residual  +  BN  +  activation
        res = self.residual(x_flat)
        out = self.act(self.bn(out + res))

        if x.dim() == 3:
            out = out.reshape(B, T, -1)

        return out


class KANModule(nn.Module):
    """
    Two-layer KAN that operates independently at each time step (Eq. 12).
    Input  : Z̃  ∈ ℝ^{T × d}
    Output : H  ∈ ℝ^{T × d_kan}

    The KAN is applied point-wise over the time axis; it does NOT model
    temporal dependencies (those are left to the Informer encoder).
    """

    def __init__(self, d_model: int, d_kan: int, num_basis: int = 10):
        super().__init__()
        self.layer1 = KANLayer(d_model, d_model, num_basis)
        self.layer2 = KANLayer(d_model, d_kan, num_basis)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, T, d_model)
        Returns:
            h : (B, T, d_kan)
        """
        h = self.layer1(z)   # (B, T, d_model)
        h = self.layer2(h)   # (B, T, d_kan)
        return h


# =============================================================================
# 2.  Discrepancy-Aware Detection  (CorDiff)  – Section 3.6
# =============================================================================

class CorDiff(nn.Module):
    """
    Computes the discrepancy D_t = KL(P_t ‖ Q_t) between:
      • P_t : global correlation distribution  (derived from attention weights)
      • Q_t : local (Gaussian prior) correlation distribution  (Eq. 19)

    Returns per-time-step KL divergence values of shape (B, T').
    """

    def __init__(self, sigma: float = 10.0):
        super().__init__()
        # σ controls the spread of the local Gaussian prior (Eq. 19)
        self.sigma = sigma

    def _local_prior(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Q_{t,j} ∝ exp( -(t-j)² / (2σ²) )   shape: (T, T)
        """
        idx = torch.arange(T, device=device, dtype=torch.float32)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)        # (T, T)
        Q = torch.exp(-diff ** 2 / (2 * self.sigma ** 2)) # (T, T)
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return Q  # (T, T)

    def forward(
        self,
        z: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z       : encoder output  (B, T', d)
            queries : query projections  (B, T', d)
            keys    : key projections    (B, T', d)
        Returns:
            D       : per-step KL divergence  (B, T')
        """
        B, T, d = z.shape

        # ---- Global correlation  P_t  (Eq. 18) ----
        scale = math.sqrt(d)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / scale  # (B, T, T)
        P = torch.softmax(scores, dim=-1)                           # (B, T, T)

        # ---- Local prior  Q_t  (Eq. 19) ----
        Q = self._local_prior(T, z.device)    # (T, T)
        Q = Q.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

        # ---- KL divergence  D_t = Σ_j P_t(j) log(P_t(j)/Q_t(j))  (Eq. 20) ----
        P_clamped = P.clamp(min=1e-9)
        Q_clamped = Q.clamp(min=1e-9)
        kl = (P_clamped * (torch.log(P_clamped) - torch.log(Q_clamped))).sum(dim=-1)
        # kl: (B, T)
        return kl   # Dt for each time step


# =============================================================================
# 3.  Multi-Scale Gated Fusion  – Section 3.7  (Eq. 24-26)
# =============================================================================

class GatedFusion(nn.Module):
    """
    Fuses KAN (local) and Informer (global) representations via a learnable gate.

        g_t  = σ( W_g [h_KAN_t ‖ h̃_Inf_t] )
        h^F_t = g_t ⊙ h_KAN_t + (1 − g_t) ⊙ h̃_Inf_t
    """

    def __init__(self, d_kan: int, d_inf: int, d_out: int):
        super().__init__()
        self.d_kan = d_kan
        self.d_inf = d_inf
        # Project both streams to d_out if dimensions differ
        self.proj_kan = nn.Linear(d_kan, d_out)
        self.proj_inf = nn.Linear(d_inf, d_out)
        # Gate
        self.gate = nn.Linear(d_out * 2, d_out)

    def forward(
        self,
        h_kan: torch.Tensor,   # (B, T, d_kan)
        h_inf: torch.Tensor,   # (B, T', d_inf)
    ) -> torch.Tensor:
        """Returns fused representation (B, T, d_out)."""
        B, T, _ = h_kan.shape

        # --- Align temporal dimensions (upsample Informer output to T) ---
        # h_inf: (B, T', d_inf)  →  (B, T, d_inf)
        h_inf_up = F.interpolate(
            h_inf.permute(0, 2, 1),   # (B, d_inf, T')
            size=T,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)            # (B, T, d_inf)

        # Project to common dimension
        h_k = self.proj_kan(h_kan)    # (B, T, d_out)
        h_i = self.proj_inf(h_inf_up) # (B, T, d_out)

        # Gating (Eq. 25-26)
        cat = torch.cat([h_k, h_i], dim=-1)    # (B, T, 2*d_out)
        g = torch.sigmoid(self.gate(cat))       # (B, T, d_out)
        h_fused = g * h_k + (1.0 - g) * h_i   # (B, T, d_out)
        return h_fused


# =============================================================================
# 4.  KAN-Informer  (full model)  – Section 3
# =============================================================================

class Model(nn.Module):
    """
    KAN-Informer for multivariate time series anomaly detection.

    Forward pass (anomaly_detection task):
        x_enc  (B, T, N)  →  reconstructed output  (B, T, N)

    Anomaly score is computed externally via:
        s_t = α · ‖x_t − x̂_t‖²  +  (1 − α) · D_t

    The model also exposes  self.cordiff_score(z, q, k)  to compute D_t
    so that training loss can incorporate Eq. (30).
    """

    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len   = configs.seq_len
        self.d_model   = configs.d_model       # embedding / Informer hidden dim
        self.c_out     = configs.c_out         # output variables  (= enc_in)

        # ---- Hyper-parameters with safe defaults ----
        num_basis  = getattr(configs, "kan_basis",  10)    # K in Eq. (11)
        d_kan      = getattr(configs, "d_kan",      configs.d_model)
        alpha      = getattr(configs, "alpha",      0.5)   # Eq. (3)
        sigma      = getattr(configs, "sigma",      10.0)  # Eq. (19)
        lam        = getattr(configs, "lam",        0.1)   # λ in Eq. (31)
        distil     = getattr(configs, "distil",     True)
        self.alpha = alpha
        self.lam   = lam

        # ------------------------------------------------------------------
        # (A) Input Embedding  –  Section 3.3
        #     Normalization is handled at forward time (feature-wise z-score).
        #     DataEmbedding performs:  linear projection  +  positional encoding
        # ------------------------------------------------------------------
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # ------------------------------------------------------------------
        # (B) KAN Module  –  Section 3.4
        # ------------------------------------------------------------------
        self.kan = KANModule(
            d_model   = configs.d_model,
            d_kan     = d_kan,
            num_basis = num_basis
        )

        # ------------------------------------------------------------------
        # (C) Informer Encoder  –  Section 3.5
        #     ProbSparse attention  +  distilling ConvLayers
        # ------------------------------------------------------------------
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            configs.factor,
                            attention_dropout = configs.dropout,
                            output_attention  = False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout    = configs.dropout,
                    activation = configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            # ConvLayer distilling reduces sequence length across layers
            [
                ConvLayer(configs.d_model)
                for _ in range(configs.e_layers - 1)
            ] if distil else None,
            norm_layer = nn.LayerNorm(configs.d_model)
        )

        # Q, K projections for CorDiff (we reuse the same linear weight as in
        # standard attention for consistency, but keep them separate here so
        # CorDiff can be called independently of the Informer internals).
        self.q_proj = nn.Linear(configs.d_model, configs.d_model, bias=False)
        self.k_proj = nn.Linear(configs.d_model, configs.d_model, bias=False)

        # ------------------------------------------------------------------
        # (D) CorDiff  –  Section 3.6
        # ------------------------------------------------------------------
        self.cordiff = CorDiff(sigma=sigma)

        # ------------------------------------------------------------------
        # (E) Gated Multi-Scale Fusion  –  Section 3.7
        # ------------------------------------------------------------------
        self.fusion = GatedFusion(
            d_kan  = d_kan,
            d_inf  = configs.d_model,
            d_out  = configs.d_model
        )

        # ------------------------------------------------------------------
        # (F) Decoder / Reconstruction  –  Section 3.8
        # ------------------------------------------------------------------
        self.projection = nn.Linear(configs.d_model, self.c_out, bias=True)

        # Dropout + LayerNorm for regularisation
        self.dropout  = nn.Dropout(configs.dropout)
        self.norm_out = nn.LayerNorm(configs.d_model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, x: torch.Tensor):
        """
        Feature-wise z-score normalisation  (Eq. 5).
        Returns (x_norm, mean, std) – mean/std needed to de-normalise later.
        """
        mu  = x.mean(dim=1, keepdim=True)                                   # (B,1,N)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt()    # (B,1,N)
        return (x - mu) / std, mu, std

    # ------------------------------------------------------------------
    # Main anomaly-detection forward
    # ------------------------------------------------------------------

    def anomaly_detection(self, x_enc: torch.Tensor, x_mark_enc=None):
        """
        Full forward pass for anomaly detection.

        Args:
            x_enc : (B, T, N)  raw multivariate time series window

        Returns:
            dec_out    : (B, T, N)   reconstructed input
            cordiff_dt : (B, T')     per-step KL discrepancy (for loss / scoring)
        """
        B, T, N = x_enc.shape

        # ---- (A) Normalisation + Embedding  (Section 3.3) ----
        x_norm, mu, std = self._normalize(x_enc)   # (B, T, N)

        # DataEmbedding: linear projection + sinusoidal positional encoding
        z = self.enc_embedding(x_norm, x_mark_enc)

        # ---- (B) KAN-based Functional Feature Learning  (Section 3.4) ----
        # Applied independently at each time step  (Eq. 12)
        h_kan = self.kan(z)                          # (B, T, d_kan)

        # ---- (C) Informer-based Sparse Temporal Modeling  (Section 3.5) ----
        # ProbSparse attention + distilling → (B, T', d_model)
        enc_out, _ = self.encoder(z, attn_mask=None) # (B, T', d_model)

        # ---- (D) CorDiff: compute Q, K from encoder output  (Section 3.6) ----
        # Eq. (18): global correlation derived from attention queries & keys
        q_enc = self.q_proj(enc_out)                 # (B, T', d_model)
        k_enc = self.k_proj(enc_out)                 # (B, T', d_model)
        cordiff_dt = self.cordiff(enc_out, q_enc, k_enc)  # (B, T')

        # ---- (E) Multi-Scale Gated Fusion  (Section 3.7) ----
        # Upsample Informer output back to T and fuse with KAN features
        h_fused = self.fusion(h_kan, enc_out)        # (B, T, d_model)
        h_fused = self.norm_out(self.dropout(h_fused))

        # ---- (F) Reconstruction  (Section 3.8) ----
        dec_out = self.projection(h_fused)           # (B, T, N)

        # De-normalise to original scale
        dec_out = dec_out * std + mu

        return dec_out, cordiff_dt

    # ------------------------------------------------------------------
    # Anomaly score computation  (Eq. 3)
    # ------------------------------------------------------------------

    def compute_anomaly_score(
        self,
        x:          torch.Tensor,   # (B, T, N)  original input
        x_hat:      torch.Tensor,   # (B, T, N)  reconstructed
        cordiff_dt: torch.Tensor,   # (B, T')    KL discrepancy
        alpha:      float = None
    ) -> torch.Tensor:
        """
        s_t = α · ‖x_t − x̂_t‖²₂  +  (1−α) · D_t          (Eq. 3)

        D_t is upsampled to length T when T' < T (due to distilling).

        Returns: anomaly_scores  (B, T)
        """
        if alpha is None:
            alpha = self.alpha

        # Reconstruction error per time step  (B, T)
        rec_err = (x - x_hat).pow(2).mean(dim=-1)

        # Upsample discrepancy to T if needed
        B, T  = rec_err.shape
        T_pri = cordiff_dt.shape[1]
        if T_pri != T:
            dt = F.interpolate(
                cordiff_dt.unsqueeze(1),    # (B, 1, T')
                size=T,
                mode="linear",
                align_corners=False
            ).squeeze(1)                    # (B, T)
        else:
            dt = cordiff_dt

        score = alpha * rec_err + (1.0 - alpha) * dt
        return score   # (B, T)

    # ------------------------------------------------------------------
    # Training loss  (Eq. 31)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x:          torch.Tensor,
        x_hat:      torch.Tensor,
        cordiff_dt: torch.Tensor
    ) -> dict:
        """
        L_total = L_rec  +  λ · L_disc          (Eq. 31)

        L_rec  = (1/T) Σ ‖x_t − x̂_t‖²          (Eq. 29)
        L_disc = (1/T') Σ KL(P_t ‖ Q_t)         (Eq. 30)

        Returns a dict with individual and total losses.
        """
        L_rec  = F.mse_loss(x_hat, x)
        L_disc = cordiff_dt.mean()
        L_total = L_rec + self.lam * L_disc
        return {"loss": L_total, "L_rec": L_rec, "L_disc": L_disc}

    # ------------------------------------------------------------------
    # Threshold-based detection  (Eq. 38)
    # ------------------------------------------------------------------

    @staticmethod
    def detect(
        scores:     torch.Tensor,   # (B, T) or (T,)
        beta:       float = 3.0,
        train_scores: torch.Tensor = None
    ) -> torch.Tensor:
        """
        τ = μ_s + β · σ_s   (Eq. 38)

        If train_scores is provided, μ_s and σ_s are computed from it;
        otherwise they are computed from scores itself.

        Returns binary labels  (B, T)  {0, 1}.
        """
        ref = train_scores if train_scores is not None else scores
        mu  = ref.mean()
        std = ref.std()
        tau = mu + beta * std
        return (scores > tau).long()

    # ------------------------------------------------------------------
    # Sliding-window score smoothing  (Eq. 21 / 36)
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_scores(scores: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        s̃_t = (1/(2k+1)) Σ_{i=t-k}^{t+k} s_i

        Operates on the last dimension.  scores: (..., T)
        """
        T = scores.shape[-1]
        pad = k
        # Reflect-pad to handle boundaries
        s_padded = F.pad(scores, (pad, pad), mode="reflect")
        kernel   = scores.new_ones(1, 1, 2 * k + 1) / (2 * k + 1)
        shape    = scores.shape
        s_flat   = s_padded.reshape(-1, 1, T + 2 * pad)
        smoothed = F.conv1d(s_flat, kernel).reshape(shape)
        return smoothed

    # ------------------------------------------------------------------
    # Unified forward  (mirrors informer.py interface)
    # ------------------------------------------------------------------

    def forward(self, x_enc, x_mark_enc = None, x_dec= None, x_mark_dec = None, mask = None):
        if self.task_name == "anomaly_detection":
            dec_out, cordiff_dt = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # (B, T, N)

        raise NotImplementedError(
            f"KAN-Informer currently supports 'anomaly_detection' only, "
            f"got task_name='{self.task_name}'."
        )

    # def forward_with_scores(
    #     self,
    #     x_enc,
    #     x_mark_enc = None
    # ):
    #     """
    #     Convenience method that returns both the reconstruction and
    #     the per-step anomaly score (for inference / evaluation).

    #     Returns:
    #         dec_out       : (B, T, N)
    #         anomaly_score : (B, T)
    #         cordiff_dt    : (B, T')
    #     """
    #     dec_out, cordiff_dt = self.anomaly_detection(x_enc)
    #     anomaly_score = self.compute_anomaly_score(x_enc, dec_out, cordiff_dt)
    #     return dec_out, anomaly_score, cordiff_dt


# # =============================================================================
# # 5.  Example training step (pseudo-code / reference)
# # =============================================================================

# def example_training_step(model, batch_x, optimizer):
#     """
#     Minimal training step following Algorithm 1 in the paper.

#     Args:
#         model   : KAN-Informer  (Model instance)
#         batch_x : (B, T, N)  multivariate time-series window (normal samples)
#         optimizer: torch.optim.Adam

#     Returns:
#         loss value (float)
#     """
#     optimizer.zero_grad()

#     # Steps 1-7 of Algorithm 1
#     dec_out, cordiff_dt = model.anomaly_detection(batch_x)

#     # Eq. 31: L_total = L_rec + λ · L_disc
#     losses = model.compute_loss(batch_x, dec_out, cordiff_dt)
#     loss   = losses["loss"]

#     loss.backward()
#     optimizer.step()

#     return loss.item()


# def example_inference(model, test_x, train_scores=None, smooth_k=3, beta=3.0):
#     """
#     Inference following Algorithm 1 Steps 8 (detection).

#     Args:
#         model        : trained KAN-Informer
#         test_x       : (B, T, N)
#         train_scores : (B_train, T) training-set anomaly scores for threshold
#         smooth_k     : window half-size for score smoothing
#         beta         : threshold scaling factor  (Eq. 38)

#     Returns:
#         labels : (B, T)  binary anomaly labels
#         scores : (B, T)  smoothed anomaly scores
#     """
#     model.eval()
#     with torch.no_grad():
#         _, raw_scores, _ = model.forward_with_scores(test_x)

#     # Optional smoothing  (Eq. 36)
#     scores = Model.smooth_scores(raw_scores, k=smooth_k)

#     # Thresholding  (Eq. 37)
#     labels = Model.detect(scores, beta=beta, train_scores=train_scores)

#     return labels, scores