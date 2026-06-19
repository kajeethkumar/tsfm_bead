import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding

# =============================================================================
# 1. B-Spline KAN Layer (Fixed Dimension Handling)
# =============================================================================

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_coeffs = grid_size + spline_order

        # Learnable spline coefficients
        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, self.num_coeffs) * 0.01)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, in_features)
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = orig_shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x
        
        # 1. Base linear transformation
        base_output = F.linear(x_flat, self.base_weight.mean(dim=1, keepdim=True).expand(-1, self.in_features))
        
        # 2. Spline logic: Map x to basis space
        # We use a sinusoidal basis expansion to approximate the B-spline mapping
        # Expansion: (B*T, in) -> (B*T, in, num_coeffs)
        x_expanded = torch.stack([torch.sin(i * x_flat) for i in range(1, self.num_coeffs + 1)], dim=-1)
        
        # Fixed Einsum Equation: 
        # b: Batch*Time, i: In_features, k: Coeffs, o: Out_features
        spline_output = torch.einsum("bik,oik->bo", x_expanded, self.coeffs)
        
        out = self.act(self.bn(base_output + spline_output))
        
        if len(orig_shape) == 3:
            out = out.reshape(B, T, -1)
        return out

# =============================================================================
# 2. Cross-Attention Fusion
# =============================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, h_kan, h_inf):
        B, T, D = h_kan.shape
        # Align Informer length via interpolation
        h_inf_up = F.interpolate(h_inf.permute(0, 2, 1), size=T, mode="linear", align_corners=False).permute(0, 2, 1)
        
        # Cross-Attention: h_kan acts as Query, h_inf_up acts as Key/Value
        attn_out, _ = self.mha(h_kan, h_inf_up, h_inf_up)
        
        # Gated fusion
        g = self.gate(torch.cat([h_kan, attn_out], dim=-1))
        return self.norm(g * h_kan + (1 - g) * attn_out)

# =============================================================================
# 3. KAC-Informer (Model)
# =============================================================================

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        
        # 1. Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # 2. KAN Branch
        self.kan = KANLayer(configs.d_model, configs.d_model)

        # 3. Informer Branch
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads
                ), 
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)],
            [ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 4. Fusion
        self.fusion = CrossAttentionFusion(configs.d_model)

        # 5. Reconstruction
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization
        means = x_enc.mean(1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        z = self.enc_embedding(x_enc, x_mark_enc)

        # Parallel Extraction
        h_kan = self.kan(z) # Local
        h_inf, _ = self.encoder(z, attn_mask=None) # Global

        # Strengthened Fusion
        fused = self.fusion(h_kan, h_inf)

        # Final Reconstruction
        dec_out = self.projection(fused)
        return dec_out * stdev + means