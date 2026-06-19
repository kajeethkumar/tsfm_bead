import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding

# =============================================================================
# 1. KAN Layer (Fourier Basis)
# =============================================================================

class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_basis: int = 10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = num_basis 

        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * 0.01
        )

        self.register_buffer(
            "freqs",
            torch.arange(1, num_basis + 1, dtype=torch.float32)
        )

        self.residual = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.GELU()

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(-1)                           
        freqs = self.freqs.view(*([1] * x.dim()), -1)     
        return torch.cos(freqs * x_exp)                   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if x.dim() == 3:
            B, T, D = shape
            x_flat = x.reshape(B * T, D)          
        else:
            x_flat = x                             

        Bx = self.basis(x_flat)                    
        phi = torch.einsum("bik,oik->boi", Bx, self.coeffs)  
        out = phi.sum(dim=-1)                      

        res = self.residual(x_flat)
        out = self.act(self.bn(out + res))

        if x.dim() == 3:
            out = out.reshape(B, T, -1)
        return out

class KANModule(nn.Module):
    def __init__(self, d_model: int, d_kan: int, num_basis: int = 10):
        super().__init__()
        self.layer1 = KANLayer(d_model, d_model, num_basis)
        self.layer2 = KANLayer(d_model, d_kan, num_basis)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(z))

# =============================================================================
# 2. Multi-Scale Gated Fusion
# =============================================================================

class GatedFusion(nn.Module):
    def __init__(self, d_kan: int, d_inf: int, d_out: int):
        super().__init__()
        self.proj_kan = nn.Linear(d_kan, d_out)
        self.proj_inf = nn.Linear(d_inf, d_out)
        self.gate = nn.Linear(d_out * 2, d_out)

    def forward(self, h_kan: torch.Tensor, h_inf: torch.Tensor) -> torch.Tensor:
        B, T, _ = h_kan.shape
        
        # Upsample Informer output (T') to original length (T)
        h_inf_up = F.interpolate(
            h_inf.permute(0, 2, 1), 
            size=T,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)            

        h_k = self.proj_kan(h_kan)    
        h_i = self.proj_inf(h_inf_up) 

        cat = torch.cat([h_k, h_i], dim=-1)    
        g = torch.sigmoid(self.gate(cat))       
        return g * h_k + (1.0 - g) * h_i   

# =============================================================================
# 3. KAN-Informer (Time-Series-Library Compatible)
# =============================================================================

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.c_out = configs.c_out

        # Hyper-parameters
        num_basis = getattr(configs, 'kan_basis', 10)
        d_kan = getattr(configs, 'd_kan', configs.d_model)
        distil = getattr(configs, 'distil', True)

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # KAN Branch
        self.kan = KANModule(
            d_model=configs.d_model,
            d_kan=d_kan,
            num_basis=num_basis
        )

        # Informer Encoder Branch
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            [
                ConvLayer(configs.d_model)
                for _ in range(configs.e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Gated Fusion
        self.fusion = GatedFusion(
            d_kan=d_kan,
            d_inf=configs.d_model,
            d_out=configs.d_model
        )

        # Reconstruction Header
        self.projection = nn.Linear(configs.d_model, self.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 1. Normalization (Feature-wise z-score)
        means = x_enc.mean(1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 2. Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (B, T, d_model)

        # 3. Parallel Processing
        h_kan = self.kan(enc_out)                       # Local Functional Features
        h_inf, _ = self.encoder(enc_out, attn_mask=None) # Global Temporal Features (B, T', d_model)

        # 4. Multi-Scale Fusion
        fused = self.fusion(h_kan, h_inf)               # (B, T, d_model)

        # 5. Projection & De-normalization
        dec_out = self.projection(fused)                # (B, T, c_out)
        dec_out = dec_out * stdev + means
        
        return dec_out  # [B, T, N]