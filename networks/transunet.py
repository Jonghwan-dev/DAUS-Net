import torch
import torch.nn as nn
import numpy as np
from .transUnet.transunet import TransUnet
from .transUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from .transUnet.vit_seg_modeling import Transformer as ViT_Transformer
from .transUnet.vit_seg_modeling import DecoderCup as ViT_Decoder
from .transUnet.vit_seg_modeling import SegmentationHead as ViT_SegHead
from .transUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class LoRALinear(nn.Module):
    """LoRA adapter around a base Linear. Trainable low-rank A/B with scaling.
    Creates parameters named with prefix 'lora_' for optimizer filtering.
    """
    def __init__(self, base: nn.Linear, r: int = 0, lora_alpha: float = 1.0, lora_dropout: float = 0.0, lora_only: bool = False):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.has_bias = base.bias is not None
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.r) if (self.r and self.r > 0) else 1.0
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()
        self.lora_only = bool(lora_only)
        if self.r and self.r > 0:
            self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A, self.lora_B = None, None
        # Optionally freeze base when only training LoRA
        if self.lora_only:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        if self.r and (self.lora_A is not None) and (self.lora_B is not None):
            lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            # Optional runtime scale (set externally)
            runtime_scale = getattr(self, 'runtime_scale', None)
            if runtime_scale is None:
                out = out + lora_out
            else:
                out = out + lora_out * runtime_scale
        return out

    def set_runtime_scale(self, scale: float):
        self.runtime_scale = float(scale)


class TransUnetTM(nn.Module):
    """
    TransUNet backbone + prompt conditioning + dual classification heads.
    - Prompts (position 8, task 2, type 3, nature 2) -> projected to hidden and added as a global bias to all patch tokens.
    - Segmentation head from TransUNet.
    - Classification heads (2-class, 4-class) from pooled encoder tokens.
    Output: (seg_logits, cls2_logits, cls4_logits)
    """
    def __init__(self, img_size=224, in_chans=1, seg_out_ch=2, hidden_size=None,
                 lora_rank: int = 0, lora_alpha: float = 1.0, lora_dropout: float = 0.0, lora_only: bool = False,
                 max_lora_scale: float = 1.0, scale_mode: str = 'sigmoid',
                 film_scale: float = 1.0, prior_lambda: float = 0.5,
                 cls_head_variant: str = 'linear', cls_dropout: float = 0.2):
        super().__init__()
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = seg_out_ch
        config_vit.n_skip = 3
        if (img_size % 16) != 0:
            raise ValueError("img_size must be divisible by 16 for TransUnet.")
        config_vit.patches.grid = (int(img_size / 16), int(img_size / 16))

        # Build components explicitly to allow prompt injection between embeddings and encoder
        self.transformer = ViT_Transformer(config_vit, img_size=img_size, vis=False)
        self.decoder = ViT_Decoder(config_vit)
        self.segmentation_head = ViT_SegHead(
            in_channels=config_vit['decoder_channels'][-1],
            out_channels=config_vit['n_classes'],
            kernel_size=3,
        )
        self.config = config_vit
        self.lora_cfg = {
            'rank': int(lora_rank),
            'alpha': float(lora_alpha),
            'dropout': float(lora_dropout),
            'only': bool(lora_only),
        }
        self.film_scale = float(film_scale)
        self.prior_lambda = float(prior_lambda)
        # Classification head options
        self.cls_head_variant = str(cls_head_variant)
        self.cls_dropout = float(cls_dropout)

        # Adapt first conv to in_chans
        if in_chans == 1 and self.transformer.embeddings.patch_embeddings.in_channels == 3:
            original_weights = self.transformer.embeddings.patch_embeddings.weight.data
            new_first_conv = nn.Conv2d(1, config_vit.hidden_size, kernel_size=self.transformer.embeddings.patch_embeddings.kernel_size,
                                       stride=self.transformer.embeddings.patch_embeddings.stride)
            new_first_conv.weight.data = original_weights.mean(dim=1, keepdim=True)
            self.transformer.embeddings.patch_embeddings = new_first_conv

        # Prompt embeddings (tokenized or one-hot compatible)
        h = config_vit.hidden_size if hidden_size is None else hidden_size
        self.pos_dim, self.task_dim, self.type_dim, self.nat_dim = 8, 2, 3, 2
        self.emb_pos = nn.Embedding(self.pos_dim, h)
        self.emb_task = nn.Embedding(self.task_dim, h)
        self.emb_type = nn.Embedding(self.type_dim, h)
        self.emb_nat = nn.Embedding(self.nat_dim, h)
        self.prompt_dropout = nn.Dropout(p=0.0)

        # Modular Prompt Controller: per-type heads -> per-layer, per-group scales
        # groups: ['q','k','v','o','fc1','fc2']
        self.scale_groups = ['q','k','v','o','fc1','fc2']
        L = len(self.transformer.encoder.layer)
        G = len(self.scale_groups)
        def make_head():
            return nn.Sequential(
                nn.Linear(h, h), nn.ReLU(inplace=True),
                nn.Linear(h, L * G)
            )
        self.ctrl_pos = make_head()
        self.ctrl_task = make_head()
        self.ctrl_type = make_head()
        self.ctrl_nat = make_head()
        # scaling mode
        self.scale_mode = scale_mode  # 'sigmoid'|'softplus'|'tanh'
        self.max_lora_scale = float(max_lora_scale)

        # Classification heads (from pooled encoder tokens)
        self.cls_norm = nn.LayerNorm(h)
        # Build heads based on variant
        def make_trunk(in_dim, d_drop):
            # 3-layer MLP: in->512->256->256 with GELU and Dropout
            return nn.Sequential(
                nn.Dropout(p=d_drop),
                nn.Linear(in_dim, 768),
                nn.GELU(),
                nn.Dropout(p=d_drop),
                nn.Linear(768, 512),
                nn.GELU(),
                nn.Dropout(p=d_drop),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Dropout(p=d_drop),
            )

        if self.cls_head_variant == 'shared_mlp':
            self.cls_trunk = make_trunk(h, self.cls_dropout)
            self.cls_head_2 = nn.Linear(512, 2)
            self.cls_head_4 = nn.Linear(512, 4)
        elif self.cls_head_variant == 'per_head_mlp':
            self.cls_trunk_2 = make_trunk(h, self.cls_dropout)
            self.cls_trunk_4 = make_trunk(h, self.cls_dropout)
            self.cls_head_2 = nn.Linear(512, 2)
            self.cls_head_4 = nn.Linear(512, 4)
        else:
            # default linear heads on pooled hidden size
            self.cls_head_2 = nn.Linear(h, 2)
            self.cls_head_4 = nn.Linear(h, 4)
        # Prompt-conditioned FiLM and prior logits
        self.film_head = nn.Linear(h, 2*h)  # -> gamma, beta
        self.prior_head_2 = nn.Linear(h, 2)
        self.prior_head_4 = nn.Linear(h, 4)

        # Apply LoRA to ViT encoder blocks if enabled
        self._maybe_apply_lora()

        # Initialize embeddings small to avoid early disruption
        for emb in [self.emb_pos, self.emb_task, self.emb_type, self.emb_nat]:
            nn.init.normal_(emb.weight, std=1e-3)

    def _maybe_apply_lora(self):
        r = self.lora_cfg['rank']
        if not r or r <= 0:
            return
        alpha = self.lora_cfg['alpha']
        dr = self.lora_cfg['dropout']
        only = self.lora_cfg['only']
        # The encoder has layers in self.transformer.encoder.layer (ModuleList of Blocks)
        # keep references for runtime scaling by group
        self._lora_modules_per_layer = []  # list of dict(group->list[LoRALinear])
        for blk in self.transformer.encoder.layer:
            attn = getattr(blk, 'attn', None)
            entry = {g: [] for g in self.scale_groups}
            if attn is not None:
                if isinstance(attn.query, nn.Linear):
                    attn.query = LoRALinear(attn.query, r=r, lora_alpha=alpha, lora_dropout=dr, lora_only=only)
                    entry['q'].append(attn.query)
                if isinstance(attn.key, nn.Linear):
                    attn.key = LoRALinear(attn.key, r=r, lora_alpha=alpha, lora_dropout=dr, lora_only=only)
                    entry['k'].append(attn.key)
                if isinstance(attn.value, nn.Linear):
                    attn.value = LoRALinear(attn.value, r=r, lora_alpha=alpha, lora_dropout=dr, lora_only=only)
                    entry['v'].append(attn.value)
                if isinstance(attn.out, nn.Linear):
                    attn.out = LoRALinear(attn.out, r=r, lora_alpha=alpha, lora_dropout=dr, lora_only=only)
                    entry['o'].append(attn.out)
            ffn = getattr(blk, 'ffn', None)
            if ffn is not None:
                if isinstance(ffn.fc1, nn.Linear):
                    ffn.fc1 = LoRALinear(ffn.fc1, r=r, lora_alpha=alpha, lora_dropout=dr, lora_only=only)
                    entry['fc1'].append(ffn.fc1)
                if isinstance(ffn.fc2, nn.Linear):
                    ffn.fc2 = LoRALinear(ffn.fc2, r=r, lora_alpha=alpha, lora_dropout=dr, lora_only=only)
                    entry['fc2'].append(ffn.fc2)
            self._lora_modules_per_layer.append(entry)

    def _encode_with_prompt(self, x, position_prompt=None, task_prompt=None, type_prompt=None, nature_prompt=None):
        # x: (B, C, H, W)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        embeddings, features = self.transformer.embeddings(x)  # (B, N, H), features: list

        if position_prompt is not None:
            # Accept either integer IDs (B,) or one-hot floats (B, dim)
            def to_ids(t, nclass):
                if t.dtype in (torch.long, torch.int64, torch.int32):
                    return t.view(-1)
                # assume one-hot or probabilities of shape (B, nclass)
                if t.dim() == 2 and t.size(-1) == nclass:
                    return t.argmax(dim=-1)
                raise ValueError(f"Unsupported prompt tensor shape {tuple(t.shape)} and dtype {t.dtype}")

            pos_ids = to_ids(position_prompt, self.pos_dim)
            task_ids = to_ids(task_prompt, self.task_dim)
            type_ids = to_ids(type_prompt, self.type_dim)
            nat_ids = to_ids(nature_prompt, self.nat_dim)

            p_pos = self.emb_pos(pos_ids)
            p_task = self.emb_task(task_ids)
            p_type = self.emb_type(type_ids)
            p_nat = self.emb_nat(nat_ids)

            # Sum embeddings and apply dropout
            p = p_pos + p_task + p_type + p_nat
            p = self.prompt_dropout(p)
            # Broadcast add to all tokens
            embeddings = embeddings + p.unsqueeze(1)

            # Prompt-controlled LoRA scaling (per-layer, per-group)
            if hasattr(self, '_lora_modules_per_layer') and len(self._lora_modules_per_layer) > 0:
                B = position_prompt.size(0)
                # compute per-type scores then sum
                def activate(x):
                    if self.scale_mode == 'sigmoid':
                        return torch.sigmoid(x)
                    elif self.scale_mode == 'softplus':
                        return torch.nn.functional.softplus(x)
                    elif self.scale_mode == 'tanh':
                        return 0.5 * (torch.tanh(x) + 1.0)
                    else:
                        return torch.sigmoid(x)
                sp = self.ctrl_pos(p_pos).view(B, -1, len(self.scale_groups))
                st = self.ctrl_task(p_task).view(B, -1, len(self.scale_groups))
                sty = self.ctrl_type(p_type).view(B, -1, len(self.scale_groups))
                sn = self.ctrl_nat(p_nat).view(B, -1, len(self.scale_groups))
                scales_bg = sp + st + sty + sn  # (B, L, G)
                scales_bg = activate(scales_bg)
                # average across batch for stability
                scales_bg = scales_bg.mean(dim=0)  # (L, G)
                scales_bg = torch.clamp(scales_bg, 0.0, self.max_lora_scale)
                # Stay in torch to avoid bf16->numpy issue under AMP; cast to float32 on CPU
                scales_bg_t = scales_bg.detach().float().cpu()
                L, G = scales_bg_t.shape
                for li, entry in enumerate(self._lora_modules_per_layer):
                    if li >= L:
                        break
                    for gi, gname in enumerate(self.scale_groups):
                        s = float(scales_bg_t[li, gi].item())
                        for m in entry[gname]:
                            m.set_runtime_scale(s)

        encoded, attn_weights = self.transformer.encoder(embeddings)
        # Also return prompt feature for contrastive alignment and classification conditioning
        prompt_feat = p if position_prompt is not None else None
        return encoded, features, prompt_feat

    def forward(self, inputs):
        # Accept either tensor image or tuple (image, pos, task, type, nature)
        if isinstance(inputs, (list, tuple)):
            image, pos_p, task_p, type_p, nat_p = inputs
            encoded, features, prompt_feat = self._encode_with_prompt(image, pos_p, task_p, type_p, nat_p)
        else:
            image = inputs
            encoded, features, prompt_feat = self._encode_with_prompt(image, None, None, None, None)

        # Segmentation path
        x_dec = self.decoder(encoded, features)
        seg_logits = self.segmentation_head(x_dec)

        # Classification from pooled encoder tokens
        pooled = encoded.mean(dim=1)
        pooled = self.cls_norm(pooled)
        # Prompt-conditioned FiLM modulation
        if prompt_feat is not None and self.film_scale > 0:
            gamma, beta = torch.chunk(self.film_head(prompt_feat), 2, dim=-1)
            gamma = torch.tanh(gamma) * self.film_scale
            beta = beta * self.film_scale
            pooled = pooled * (1.0 + gamma) + beta
        # Optional MLP trunks
        if self.cls_head_variant == 'shared_mlp':
            z = self.cls_trunk(pooled)
            img_logits_2 = self.cls_head_2(z)
            img_logits_4 = self.cls_head_4(z)
        elif self.cls_head_variant == 'per_head_mlp':
            z2 = self.cls_trunk_2(pooled)
            z4 = self.cls_trunk_4(pooled)
            img_logits_2 = self.cls_head_2(z2)
            img_logits_4 = self.cls_head_4(z4)
        else:
            img_logits_2 = self.cls_head_2(pooled)
            img_logits_4 = self.cls_head_4(pooled)
        # Prompt prior logits and combination
        if prompt_feat is not None and self.prior_lambda > 0:
            prior2 = self.prior_head_2(prompt_feat)
            prior4 = self.prior_head_4(prompt_feat)
            cls2 = img_logits_2 + self.prior_lambda * prior2
            cls4 = img_logits_4 + self.prior_lambda * prior4
        else:
            cls2 = img_logits_2
            cls4 = img_logits_4
        return seg_logits, cls2, cls4, pooled, (prompt_feat if prompt_feat is not None else pooled.detach()*0)

    # Compatibility helpers for existing training script
    def load_from_self(self, weights_npz_path):
        # Not applicable in this wrapper; keep for API compatibility
        return

    def load_from(self, config=None):
        # No-op: We rely on default initialization or external checkpoints
        return
