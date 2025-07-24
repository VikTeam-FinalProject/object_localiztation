import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionReweightAndGate(nn.Module):
    def __init__(self, vit_model, patch_size, α_init=0.5):
        super().__init__()
        self.vit = vit_model
        self.patch_size = patch_size
        L = len(vit_model.blocks)
        H = vit_model.num_heads      # adjust to your ViT implementation
        D_head = vit_model.embed_dim // H

        # 1️⃣ Per-layer, per-head weights α_{l,h}
        self.alpha = nn.Parameter(torch.full((L, H), α_init))

        # 2️⃣ Gating MLP on each patch key vector
        self.gate_mlp = nn.Sequential(
            nn.Linear(D_head, D_head),
            nn.ReLU(),
            nn.Linear(D_head, 1),
            nn.Sigmoid()
        )

        # storage for each block's raw QKV
        self._qkvs = [None] * L
        for i, blk in enumerate(self.vit.blocks):
            # hook into the qkv projection of each block
            blk.attn.qkv.register_forward_hook(self._make_hook(i))

    def _make_hook(self, idx):
        def hook(module, inp, out):
            # out shape: (B, N_tokens, 3*H*D_head)
            self._qkvs[idx] = out
        return hook

    def forward(self, x):
        # pass once through ViT to fill self._qkvs
        _ = self.vit(x)

        all_layer_scores = []
        for l, qkv in enumerate(self._qkvs):
            print("qkv shape:", qkv.shape)
            B, N_tokens, total_dim = qkv.shape
            H      = self.vit.num_heads
            D_head = total_dim // (3 * H)

            # reshape into (3, B, H, N_tokens, D_head)
            qkv = qkv.view(B, N_tokens, 3, H, D_head) \
                    .permute(2, 0, 3, 1, 4)

            # split
            q = qkv[0]  # (B,H,N_tokens,D_head)
            k = qkv[1]
            # v = qkv[2]  # if you need it

            # gate on K
            gates = self.gate_mlp(k[:, :, 1:, :])       # skip CLS
            keep  = (gates.squeeze(-1) > 0.5)           # (B,H,N_tokens-1)
            q_clone = q.clone()
            q_clone[:, :, 1:, :][~keep] = 0.0

            # CLS vs patches
            Q_cls     = q_clone[:, :, 0:1, :]           # (B,H,1,D_head)
            Q_patches = q_clone[:, :, 1:, :]            # (B,H,Np,D_head)

            # head‑wise scores
            head_scores = torch.einsum("b h d, b h n d -> b h n",
                                    Q_cls.squeeze(2),
                                    Q_patches)
            head_scores = head_scores / math.sqrt(D_head)
            head_scores = F.softmax(head_scores, dim=-1)  # (B,H,Np)

            # weight heads by α_l
            α_l = torch.softmax(self.alpha[l], dim=0)     # (H,)
            weighted = (α_l[None, :, None] * head_scores).sum(dim=1)  # (B,Np)

            all_layer_scores.append(weighted[0])  # drop the batch dim if B=1

        # fuse across layers
        fused = torch.stack(all_layer_scores).mean(dim=0)  # (Np,)
        return fused

        # for l, qkv in enumerate(self._qkvs):
        #     print("qkv shape:", qkv.shape)
        #     #B, Nt3HD = qkv.shape
        #     B, N_tokens, total_dim = qkv.shape
        #     H = self.vit.num_heads
        #     # D_head = Nt3HD // (3*H)
        #     D_head = total_dim // (3 * H)   

        #     # reshape into (3, B, H, N_tokens, D_head)
        #     qkv = qkv.view(B, N_tokens, 3, H, D_head).permute(2, 0, 3, 1, 4)
        #     #q, k, _ = qkv[0][0], qkv[1][0], qkv[2][0]
        #     # q, k: shape = (H, N_tokens, D_head)

        #     q, k = qkv[0], qkv[1]  # (H, B, N_tokens, D_head)

        #     # 2️⃣ Gate out noisy patches
        #     gates = self.gate_mlp(k[:, :, 1:, :])               
        #     keep = (gates.squeeze(-1) > 0.5)   # boolean mask
        #     # zero‑out q for dropped patches
        #     q_clone = q.clone()
        #     q_clone[:, :, 1:, :][~keep] = 0.0 

        #     # compute attention from CLS → all patches
        #     Q_cls    = q[:, :, 0:1, :]     # (B, H, 1, D_head)
        #     Q_patches = q[:, :, 1:, :]                # (H, Np, D)
        #     scores = torch.einsum("b h d, b h n d -> b h n",
        #               Q_cls.squeeze(2),
        #               Q_patches)
        #     scores = scores / math.sqrt(D_head)
        #     scores = F.softmax(scores, dim=-1)

        #     # weight heads by α_l
        #     α_l = torch.softmax(self.alpha[l], dim=0)  # (H,)
        #     weighted = (α_l[:, None] * scores).sum(dim=0)  # (Np,)

        #     all_layer_scores.append(weighted)

        # # 1️⃣ Fuse across layers (simple mean here)
        # fused = torch.stack(all_layer_scores, dim=0).mean(dim=0)  # (Np,)

        # return fused

###