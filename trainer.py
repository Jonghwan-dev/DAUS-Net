import os
import math
import sys
import warnings
import random
import logging
import datetime
import math
import copy
import numpy as np
from tqdm import tqdm

# Suppress noisy warnings
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument",
    category=UserWarning,
)

# Suppress CUDA/C++ backend verbose logs (e.g., DDP reducer warnings)
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
from torch.amp import autocast, GradScaler
import torch.optim as optim
import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from util.utils import DiceLovaszLoss, FocalLoss, BCEWithLogitsDiceLoss
from util.utils import FocalLovaszHingeLoss, FocalTverskyLoss
from datasets.dataset import USdatasetCls, USdatasetSeg
from datasets.omni_dataset import WeightedRandomSamplerDDP
from datasets.omni_dataset import USdatasetOmni_cls, USdatasetOmni_seg
from datasets.dataset import RandomGeneratorTM, ResizePadTM
from sklearn.metrics import roc_auc_score
from util.utils import omni_seg_test
import cv2

try:
    import wandb
    _wandb_available = True
except Exception:
    _wandb_available = False


def _compute_resize_pad_params(orig_h, orig_w, out_h, out_w):
    scale = min(out_h / orig_h, out_w / orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    off_y = (out_h - new_h) // 2
    off_x = (out_w - new_w) // 2
    return new_h, new_w, off_y, off_x


def _restore_mask_from_padded(pred_mask_hw, orig_h, orig_w, out_h, out_w):
    new_h, new_w, off_y, off_x = _compute_resize_pad_params(orig_h, orig_w, out_h, out_w)
    inner = pred_mask_hw[off_y:off_y+new_h, off_x:off_x+new_w]
    restored = cv2.resize(inner.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return restored


def omni_train_tm(args, model, snapshot_path):
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=7200))

    is_master = int(os.environ["LOCAL_RANK"]) == 0
    if is_master:
        print('** GPU NUM ** : ', torch.cuda.device_count())
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
    print(f"** DDP ** : Start running on rank {rank}.")

    logging.basicConfig(filename=snapshot_path + "/log_tm.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = getattr(args, 'base_lr', 1e-3)
    batch_size = args.batch_size
    # AMP config (prefer bf16 when available)
    amp_dtype_arg = getattr(args, 'amp_dtype', 'bf16')
    if amp_dtype_arg == 'fp16':
        _amp_dtype = torch.float16
    else:
        # default to bf16
        _amp_dtype = torch.bfloat16

    # WandB init (master only)
    use_wandb = is_master and (not getattr(args, 'wandb_off', False)) and _wandb_available
    if use_wandb:
        wandb.init(project=getattr(args, 'wandb_project', 'uusic25_tm'),
                   name=getattr(args, 'wandb_run', None),
                   entity=getattr(args, 'wandb_entity', None),
                   config={k: getattr(args, k) for k in vars(args) if not k.startswith('_')})

    def worker_init_fn(worker_id):
        # Seed Python, NumPy, and Torch for each worker for reproducibility
        base = int(args.seed)
        wid = int(worker_id)
        try:
            import numpy as _np
        except Exception:
            _np = None
        seed_val = base + wid
        random.seed(seed_val)
        if _np is not None:
            _np.random.seed(seed_val)
        try:
            torch.manual_seed(seed_val)
            torch.cuda.manual_seed(seed_val)
            torch.cuda.manual_seed_all(seed_val)
        except Exception:
            pass

    # Datasets with TM transforms (aspect ratio preserving resize + pad)
    db_train_seg = USdatasetOmni_seg(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([
            RandomGeneratorTM(
                output_size=[args.img_size, args.img_size],
                hflip_p=0.3, rot_p=0.3, max_rot_deg=10, scale_jitter=(0.95, 1.05),
                intensity_jitter=0.05, gaussian_noise_std=0.0, blur_p=0.0
            )
        ]),
        prompt=args.prompt,
    )

    if is_master:
        logging.info(f"[DBG] SEG train total samples: {len(db_train_seg)}; per-subset lens: {db_train_seg.subset_len}")

    weight_base = [
        0.25,  # 1312
        1,     # 452
        1,     # 393
        1,     # 350
        1,     # 326
        0.5,   # 699
        1,     # 340
        3,     # 105
        2,     # 165
        4,     # 53
        4,     # 84
        4,     # 46
        1,     # 299
    ]
    sample_weight_seq = [[weight_base[dataset_index]] * element for dataset_index, element in enumerate(db_train_seg.subset_len)]
    sample_weight_seq = [element for sublist in sample_weight_seq for element in sublist]

    weighted_sampler_seg = WeightedRandomSamplerDDP(
        data_set=db_train_seg,
        weights=sample_weight_seq,
        num_replicas=world_size,
        rank=rank,
        num_samples=len(db_train_seg),
        replacement=True,
    )
    trainloader_seg = DataLoader(
        db_train_seg,
        batch_size=batch_size,
        num_workers=32,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        sampler=weighted_sampler_seg,
    )

    db_train_cls = USdatasetOmni_cls(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([
            RandomGeneratorTM(
                output_size=[args.img_size, args.img_size],
                hflip_p=0.3, rot_p=0.3, max_rot_deg=10, scale_jitter=(0.95, 1.05),
                intensity_jitter=0.05, gaussian_noise_std=0.0, blur_p=0.0
            )
        ]),
        prompt=args.prompt,
        del_outlier=getattr(args, 'del_outlier', False),
    )
    if is_master:
        logging.info(f"[DBG] CLS train total samples: {len(db_train_cls)}; per-subset lens: {db_train_cls.subset_len}")

    weight_base = [
        1,     # 331
        0.25,  # 1312
        1,     # 452
        1,     # 385
        4,     # 46
        3,     # 105
        2,     # 165
        4,     # 72
    ]
    sample_weight_seq = [[weight_base[dataset_index]] * element for dataset_index, element in enumerate(db_train_cls.subset_len)]
    sample_weight_seq = [element for sublist in sample_weight_seq for element in sublist]

    weighted_sampler_cls = WeightedRandomSamplerDDP(
        data_set=db_train_cls,
        weights=sample_weight_seq,
        num_replicas=world_size,
        rank=rank,
        num_samples=len(db_train_cls),
        replacement=True,
    )
    trainloader_cls = DataLoader(
        db_train_cls,
        batch_size=batch_size,
        num_workers=32,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        sampler=weighted_sampler_cls,
    )

    if is_master:
        logging.info(f"[DBG] Train seg iters/epoch: {len(trainloader_seg)}; cls iters/epoch: {len(trainloader_cls)}")

    model = model.to(device=device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)

    model.train()

    # Segmentation criterion selection
    seg_bce_w = float(getattr(args, 'seg_bce_weight', 0.5))
    seg_dice_w = float(getattr(args, 'seg_dice_weight', 0.5))
    seg_auto_pos = bool(getattr(args, 'seg_auto_pos_weight', True))
    sel = getattr(args, 'seg_loss', 'bce_dice')
    if sel == 'lovasz':
        seg_crit = DiceLovaszLoss()
    elif sel == 'focal_lovasz':
        seg_crit = FocalLovaszHingeLoss(
            focal_weight=0.4,
            lovasz_weight=0.6,
            gamma=getattr(args, 'seg_focal_gamma', 2.0),
            alpha_fg=getattr(args, 'seg_focal_alpha_fg', 0.8),
            per_image=True,
        )
    elif sel == 'focal_tversky':
        seg_crit = FocalTverskyLoss(
            alpha=getattr(args, 'seg_tversky_alpha', 0.7),
            beta=getattr(args, 'seg_tversky_beta', 0.3),
            gamma=getattr(args, 'seg_tversky_gamma', 0.8),
        )
    else:
        # BCEWithLogits + Dice on foreground logit
        seg_crit = BCEWithLogitsDiceLoss(
            bce_weight=seg_bce_w,
            dice_weight=seg_dice_w,
            auto_pos_weight=seg_auto_pos,
            use_fg_channel=True,
        )

    # Classification criteria (configurable)
    if hasattr(args, 'cls_loss') and args.cls_loss == 'focal':
        cls_crit_2 = FocalLoss()
        cls_crit_4 = FocalLoss()
    elif hasattr(args, 'cls_loss') and args.cls_loss == 'ls':
        smoothing = getattr(args, 'label_smoothing', 0.1)
        cls_crit_2 = CrossEntropyLoss(label_smoothing=smoothing)
        cls_crit_4 = CrossEntropyLoss(label_smoothing=smoothing)
    else:
        cls_crit_2 = CrossEntropyLoss()
        cls_crit_4 = CrossEntropyLoss()

    # Param groups with head/prompt LR multiplier, or LoRA-only optimization
    head_mult = getattr(args, 'head_lr_mult', 1.0)
    lora_only_train = bool(getattr(args, 'lora_only', False) or getattr(args, 'lora_only_train', False))

    # First, collect LoRA params by name convention
    lora_params = []
    base_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_l = n.lower()
        if 'lora_' in name_l:
            lora_params.append(p)
            continue
        if ('prompt' in name_l) or ('ctrl_' in name_l) or ('emb_' in name_l) or ('head' in name_l) or ('classifier' in name_l) or ('cls' in name_l and 'norm' not in name_l) or ('seg' in name_l and 'norm' not in name_l):
            head_params.append(p)
        else:
            base_params.append(p)
    if len(base_params) == 0 and not lora_only_train:
        # Fallback if all are considered head params
        base_params = head_params
        head_params = []

    param_groups = []
    if lora_only_train:
        if len(lora_params) == 0:
            logging.warning('[LoRA] lora_only training requested but no LoRA params found; defaulting to all trainable params')
            # fall back to base/head grouping
            if base_params:
                param_groups.append({'params': base_params, 'lr': base_lr, 'initial_lr': base_lr})
            if head_params:
                param_groups.append({'params': head_params, 'lr': base_lr * head_mult, 'initial_lr': base_lr * head_mult})
        else:
            param_groups.append({'params': lora_params, 'lr': base_lr, 'initial_lr': base_lr})
            logging.info(f"[LoRA] Optimizing only LoRA params: {len(lora_params)} tensors")
    else:
        if base_params:
            param_groups.append({'params': base_params, 'lr': base_lr, 'initial_lr': base_lr})
        if head_params:
            param_groups.append({'params': head_params, 'lr': base_lr * head_mult, 'initial_lr': base_lr * head_mult})

    optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=0.05, betas=(0.9, 0.999))
    # Optional homoscedastic uncertainty weighting across tasks (seg, cls2, cls4)
    use_uncertainty_weight = bool(getattr(args, 'use_uncertainty_weight', True))
    if use_uncertainty_weight:
        # log variance parameters (initialized to 0 => weight 1.0)
        # seg, cls2, cls4, align
        log_vars = torch.nn.Parameter(torch.zeros(4, device=device))
        optimizer.add_param_group({'params': [log_vars], 'lr': base_lr, 'weight_decay': 0.0, 'initial_lr': base_lr})
    else:
        log_vars = None

    # Prompt-pooled alignment loss (InfoNCE with temperature)
    def prompt_align_loss(pooled_feat, prompt_feat, temp: float = 0.07):
        # pooled_feat, prompt_feat: (B, H)
        if (pooled_feat is None) or (prompt_feat is None):
            return pooled_feat.new_tensor(0.0)
        # Ensure same dtype/device (avoid AMP bf16/float mismatch)
        z1 = torch.nn.functional.normalize(pooled_feat.float(), dim=1)
        z2 = torch.nn.functional.normalize(prompt_feat.float(), dim=1)
        z2 = z2.to(device=z1.device, dtype=z1.dtype)
        logits = (z1 @ z2.t()) / float(temp)  # (B,B)
        targets = torch.arange(z1.size(0), device=z1.device)
        return torch.nn.functional.cross_entropy(logits, targets)

    resume_epoch = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        resume_epoch = ckpt['epoch']

    writer = SummaryWriter(snapshot_path + '/log_tm')
    use_amp = bool(getattr(args, 'amp', True))
    scaler = GradScaler(enabled=use_amp, init_scale=2.**16, growth_interval=2000)
    global_iter_num = 0
    seg_iter_num = 0
    cls_iter_num = 0
    max_epoch = args.max_epochs
    total_iterations = (len(trainloader_seg) + len(trainloader_cls))
    max_iterations = args.max_epochs * total_iterations
    logging.info("{} batch size. {} iterations per epoch. {} max iterations ".format(batch_size, total_iterations, max_iterations))
    best_performance = 0.0
    best_epoch = 0

    # EMA state
    use_ema = getattr(args, 'use_ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.999)
    ema_state = None

    # Loss ramping for classification
    def get_w_cls(ep):
        w0 = getattr(args, 'w_cls_start', 1.0)
        w1 = getattr(args, 'w_cls_end', 1.0)
        ramp = max(1, getattr(args, 'w_cls_ramp_epochs', 1))
        if ep >= ramp:
            return w1
        return w0 + (w1 - w0) * (ep / ramp)

    # Prompt projection regularization (L2 on projection weights)
    prompt_reg_coeff = float(getattr(args, 'prompt_reg', 0.0))
    def compute_prompt_reg():
        if prompt_reg_coeff <= 0.0:
            return 0.0
        reg = 0.0
        try:
            mm = model.module if hasattr(model, 'module') else model
            for name in ['proj_pos', 'proj_task', 'proj_type', 'proj_nat']:
                if hasattr(mm, name):
                    mod = getattr(mm, name)
                    if hasattr(mod, 'weight') and mod.weight is not None:
                        reg = reg + (mod.weight.pow(2).mean())
        except Exception:
            return 0.0
        return prompt_reg_coeff * reg

    # LR schedule: warmup + cosine per-iter; base amplitude controlled by current_base_lr (subject to plateau)
    current_base_lr = float(base_lr)
    warmup_epochs = int(getattr(args, 'warmup_epochs', 5))
    min_lr_ratio = float(getattr(args, 'min_lr_ratio', 0.05))
    def compute_lr(epoch, it_in_epoch, iters_per_epoch):
        # total progress in epochs (fractional)
        progress = epoch + (it_in_epoch / max(1, iters_per_epoch))
        if progress < warmup_epochs:
            warmup_factor = progress / max(1e-6, float(warmup_epochs))
            lr = current_base_lr * max(0.0, min(1.0, warmup_factor))
        else:
            # cosine from current_base_lr to current_base_lr * min_lr_ratio over remaining epochs
            T_total = float(max_epoch - warmup_epochs)
            t = max(0.0, min(T_total, progress - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * (t / max(1e-6, T_total))))
            lr = current_base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
        return max(1e-8, float(lr))

    # Early stopping / LR plateau state
    best_val = float('inf')  # lower is better for loss
    best_score = -1e9       # for early_stop_metric (higher is better)
    no_improve_epochs = 0
    stop_training = False

    if not is_master:
        iterator = tqdm(range(resume_epoch, max_epoch), ncols=70, disable=True)
    else:
        iterator = tqdm(range(resume_epoch, max_epoch), ncols=70, disable=False)

    # Helper to ensure prompt tensors have shape (B, dim)
    def _to_bdim(x, dim_expected: int, bsz: int) -> torch.Tensor:
        if x is None:
            return x
        # Accept lists/ndarrays
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x))
        if x.ndim == 1:
            # Single vector -> (1, dim)
            if x.numel() == dim_expected:
                x = x.view(1, dim_expected)
        elif x.ndim == 2:
            b0, b1 = x.shape
            # Already (B, dim)
            if b0 == bsz and b1 == dim_expected:
                pass
            # Transposed (dim, B)
            elif b0 == dim_expected and b1 == bsz:
                x = x.permute(1, 0)
            # Collated oddly: try reshape by numel
            elif x.numel() == bsz * dim_expected:
                x = x.contiguous().view(bsz, dim_expected)
        else:
            # Fallback: flatten all but batch if possible
            try:
                x = x.view(bsz, -1)
                if x.shape[1] != dim_expected and x.numel() == bsz * dim_expected:
                    x = x.view(bsz, dim_expected)
            except Exception:
                pass
        return x

    for epoch_num in iterator:
        logging.info("\n epoch: {}".format(epoch_num))
        weighted_sampler_seg.set_epoch(epoch_num)
        weighted_sampler_cls.set_epoch(epoch_num)

        # Per-epoch debug counters
        seg_imgs_epoch = 0
        cls_imgs_epoch = 0

        torch.cuda.empty_cache()
        for i_batch, sampled_batch in tqdm(enumerate(trainloader_seg), disable=not is_master):
            bsz = sampled_batch['image'].size(0)
            seg_imgs_epoch += bsz

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)
            # Optional: skip/downweight bg-only patches to avoid background collapse
            seg_skip_prob = float(getattr(args, 'seg_skip_bg_only_prob', 0.0))
            if seg_skip_prob > 0.0:
                if (label_batch > 0).sum().item() == 0:
                    if random.random() < seg_skip_prob:
                        # still advance LR schedule and global_iter for consistency (use unified epoch timeline)
                        lr_ = compute_lr(epoch_num, i_batch, (len(trainloader_seg) + len(trainloader_cls)))
                        for param_group in optimizer.param_groups:
                            mult = (param_group.get('initial_lr', base_lr) / base_lr)
                            param_group['lr'] = lr_ * mult
                        seg_iter_num += 1
                        global_iter_num += 1
                        continue

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=_amp_dtype, enabled=use_amp):
                if args.prompt:
                    position_prompt = _to_bdim(sampled_batch['position_prompt'], 8, bsz).float().to(device=device)
                    task_prompt = _to_bdim(sampled_batch['task_prompt'], 2, bsz).float().to(device=device)
                    type_prompt = _to_bdim(sampled_batch['type_prompt'], 3, bsz).float().to(device=device)
                    nature_prompt = _to_bdim(sampled_batch['nature_prompt'], 2, bsz).float().to(device=device)
                    (x_seg, _, _, pooled_feat, prompt_feat) = model((image_batch, position_prompt, task_prompt, type_prompt, nature_prompt))
                else:
                    (x_seg, _, _, pooled_feat, prompt_feat) = model(image_batch)
                # Expect logits with C==1 or C==2; ensure labels are 0/1 float
                if label_batch.dtype != torch.float32:
                    label_batch = label_batch.float()
                if label_batch.max() > 1.0:
                    label_batch = (label_batch > 0).float()
                seg_loss = seg_crit(x_seg, label_batch)
                # --- Extra constraints to suppress background expansion ---
                # Get foreground probability map
                if x_seg.shape[1] == 1:
                    fg_logit = x_seg[:, 0]
                else:
                    fg_logit = x_seg[:, 1]
                fg_prob = torch.sigmoid(fg_logit)
                gt_fg = (label_batch > 0).float().squeeze(1) if label_batch.ndim == 4 else (label_batch > 0).float()
                # Area prior: encourage predicted area to match GT area
                area_pred = fg_prob.sum(dim=(1, 2))
                area_gt = gt_fg.sum(dim=(1, 2))
                area_loss = torch.mean(torch.abs(area_pred - area_gt) / (fg_prob.shape[-1] * fg_prob.shape[-2]))
                # Hard FP top-k penalty: focus on worst background pixels
                w_fp = float(getattr(args, 'seg_fp_weight', 0.2))
                topk_frac = float(getattr(args, 'seg_fp_topk', 0.1))
                if w_fp > 0 and topk_frac > 0:
                    bg_mask = (gt_fg == 0).float()
                    # Avoid empty
                    if bg_mask.sum() > 0:
                        fp_scores = fg_prob * bg_mask
                        k = max(1, int(topk_frac * fp_scores[0].numel()))
                        fp_topk = torch.topk(fp_scores.view(fp_scores.size(0), -1), k=k, dim=1, largest=True).values
                        fp_penalty = fp_topk.mean()
                    else:
                        fp_penalty = fg_prob.new_tensor(0.0)
                else:
                    fp_penalty = fg_prob.new_tensor(0.0)
                w_area = float(getattr(args, 'seg_area_weight', 0.1))
                seg_loss = seg_loss + w_area * area_loss + w_fp * fp_penalty
                align_loss = prompt_align_loss(pooled_feat, prompt_feat)
                # Combine with uncertainty and prompt regularization
                total_loss = seg_loss + align_loss * float(getattr(args, 'w_align', 0.1))
                if use_uncertainty_weight and (log_vars is not None):
                    total_loss = (0.5 * torch.exp(-log_vars[0]) * seg_loss + 0.5 * log_vars[0]) 
                    total_loss = total_loss + (0.5 * torch.exp(-log_vars[3]) * align_loss + 0.5 * log_vars[3])
                reg_term = compute_prompt_reg()
                if isinstance(reg_term, torch.Tensor):
                    total_loss = total_loss + reg_term
                else:
                    total_loss = total_loss + float(reg_term)

            if use_amp:
                scaler.scale(total_loss).backward()
                if getattr(args, 'clip_grad_norm', 0.0) and args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if getattr(args, 'clip_grad_norm', 0.0) and args.clip_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            # EMA update (track only floating-point tensors)
            if use_ema:
                with torch.no_grad():
                    if ema_state is None:
                        ema_state = {k: v.detach().clone() for k, v in model.module.state_dict().items() if torch.is_floating_point(v)}
                    else:
                        msd = model.module.state_dict()
                        for k in list(ema_state.keys()):
                            if k not in msd:
                                continue
                            v = msd[k]
                            if not torch.is_floating_point(v):
                                continue
                            ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=(1.0 - ema_decay))

            # scheduler: unified timeline within epoch
            lr_ = compute_lr(epoch_num, i_batch, (len(trainloader_seg) + len(trainloader_cls)))
            for param_group in optimizer.param_groups:
                mult = (param_group.get('initial_lr', base_lr) / base_lr)
                param_group['lr'] = lr_ * mult

            seg_iter_num += 1
            global_iter_num += 1

            writer.add_scalar('info/lr', lr_, seg_iter_num)
            writer.add_scalar('info/seg_loss', seg_loss, seg_iter_num)
            if use_uncertainty_weight and (log_vars is not None):
                writer.add_scalar('info/log_var_seg', log_vars[0].detach().item(), seg_iter_num)
                writer.add_scalar('info/log_var_align', log_vars[3].detach().item(), seg_iter_num)
            if is_master and i_batch % 10 == 0:
                logging.info('[DBG] train seg i=%d, bsz=%d, cum_imgs=%d, seg_loss=%.4f' % (i_batch, bsz, seg_imgs_epoch, seg_loss.item()))
                if use_wandb:
                    wandb.log({'train/seg_loss': float(seg_loss.item()), 'train/lr': float(lr_), 'train/iter': global_iter_num}, step=global_iter_num)

        torch.cuda.empty_cache()
        w_cls = get_w_cls(epoch_num)
        for i_batch, sampled_batch in tqdm(enumerate(trainloader_cls), disable=not is_master):
            bsz = sampled_batch['image'].size(0)
            cls_imgs_epoch += bsz

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            num_classes_batch = sampled_batch['num_classes']
            image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)
            mixed_images = image_batch

            with autocast(device_type='cuda', dtype=_amp_dtype, enabled=use_amp):
                if args.prompt:
                    position_prompt = _to_bdim(sampled_batch['position_prompt'], 8, bsz).float().to(device=device)
                    task_prompt = _to_bdim(sampled_batch['task_prompt'], 2, bsz).float().to(device=device)
                    type_prompt = _to_bdim(sampled_batch['type_prompt'], 3, bsz).float().to(device=device)
                    nature_prompt = _to_bdim(sampled_batch['nature_prompt'], 2, bsz).float().to(device=device)
                    (_, x_cls_2, x_cls_4, pooled_feat, prompt_feat) = model((mixed_images, position_prompt, task_prompt, type_prompt, nature_prompt))
                else:
                    (_, x_cls_2, x_cls_4, pooled_feat, prompt_feat) = model(mixed_images)

            loss = 0.0
            mask_2_way = (num_classes_batch == 2)
            mask_4_way = (num_classes_batch == 4)
            # Initialize optionals for safe use below
            loss_ce_2 = None
            loss_ce_4 = None
            labels_2_way = None
            labels_4_way = None
            if mask_2_way.any():
                outputs_2_way = x_cls_2[mask_2_way]
                labels_2_way = label_batch[mask_2_way]
                # OHEM-style weighting
                import torch.nn.functional as F
                with torch.no_grad():
                    probs = F.softmax(outputs_2_way, dim=1)
                    p_true = probs.gather(1, labels_2_way.long().view(-1,1)).squeeze(1)
                    difficulty = 1.0 - p_true
                gamma = float(getattr(args, 'cls_hard_weight_gamma', 2.0))
                weights = (difficulty + 1e-6) ** gamma
                ce_per = F.cross_entropy(outputs_2_way, labels_2_way.long(), reduction='none')
                # optional top-k selection
                ohem_frac = float(getattr(args, 'cls_ohem_frac', 0.0))
                if ohem_frac > 0.0:
                    k = max(1, int(ohem_frac * ce_per.numel()))
                    vals, idxs = torch.topk(ce_per, k=k, dim=1, largest=True)
                    loss_ce_2 = (vals * weights[idxs]).mean()
                else:
                    loss_ce_2 = (ce_per * weights).mean()
                loss += loss_ce_2
            if mask_4_way.any():
                outputs_4_way = x_cls_4[mask_4_way]
                labels_4_way = label_batch[mask_4_way]
                import torch.nn.functional as F
                with torch.no_grad():
                    probs = F.softmax(outputs_4_way, dim=1)
                    p_true = probs.gather(1, labels_4_way.long().view(-1,1)).squeeze(1)
                    difficulty = 1.0 - p_true
                gamma = float(getattr(args, 'cls_hard_weight_gamma', 2.0))
                weights = (difficulty + 1e-6) ** gamma
                ce_per = F.cross_entropy(outputs_4_way, labels_4_way.long(), reduction='none')
                ohem_frac = float(getattr(args, 'cls_ohem_frac', 0.0))
                if ohem_frac > 0.0:
                    k = max(1, int(ohem_frac * ce_per.numel()))
                    vals, idxs = torch.topk(ce_per, k=k, dim=1, largest=True)
                    loss_ce_4 = (vals * weights[idxs]).mean()
                else:
                    loss_ce_4 = (ce_per * weights).mean()
                loss += loss_ce_4

            # apply task balancing weight to classification
            cls_raw = loss
            # compute alignment loss on pooled/prompt features
            align_loss = prompt_align_loss(pooled_feat, prompt_feat)
            if use_uncertainty_weight and (log_vars is not None):
                total_cls = torch.zeros((), device=device)
                if loss_ce_2 is not None:
                    total_cls = total_cls + (0.5 * torch.exp(-log_vars[1]) * loss_ce_2 + 0.5 * log_vars[1])
                if loss_ce_4 is not None:
                    total_cls = total_cls + (0.5 * torch.exp(-log_vars[2]) * loss_ce_4 + 0.5 * log_vars[2])
                total_cls = total_cls + (0.5 * torch.exp(-log_vars[3]) * align_loss + 0.5 * log_vars[3])
                loss = w_cls * total_cls
            else:
                loss = w_cls * cls_raw + float(getattr(args, 'w_align', 0.1)) * align_loss

            # add prompt projection regularization
            reg_term = compute_prompt_reg()
            if isinstance(reg_term, torch.Tensor):
                loss = loss + reg_term
            else:
                loss = loss + float(reg_term)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if getattr(args, 'clip_grad_norm', 0.0) and args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if getattr(args, 'clip_grad_norm', 0.0) and args.clip_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            # scheduler: simple per-epoch decay
            lr_ = compute_lr(epoch_num, len(trainloader_seg) + i_batch, (len(trainloader_seg) + len(trainloader_cls)))
            for param_group in optimizer.param_groups:
                mult = (param_group.get('initial_lr', base_lr) / base_lr)
                param_group['lr'] = lr_ * mult

            cls_iter_num += 1
            global_iter_num += 1

            writer.add_scalar('info/lr', lr_, cls_iter_num)
            writer.add_scalar('info/cls_loss', loss, cls_iter_num)
            if is_master and i_batch % 10 == 0:
                logging.info('[DBG] train cls i=%d, bsz=%d, cum_imgs=%d, loss=%.4f' % (i_batch, bsz, cls_imgs_epoch, loss.item()))
                if use_wandb:
                    wandb.log({'train/cls_loss': float(loss.item()), 'train/lr': float(lr_), 'train/iter': global_iter_num}, step=global_iter_num)
            if use_uncertainty_weight and (log_vars is not None):
                writer.add_scalar('info/log_var_cls2', log_vars[1].detach().item(), cls_iter_num)
                writer.add_scalar('info/log_var_cls4', log_vars[2].detach().item(), cls_iter_num)
                writer.add_scalar('info/log_var_align', log_vars[3].detach().item(), cls_iter_num)

        # End of epoch summary
        if is_master:
            logging.info(f"[DBG] EPOCH {epoch_num} summary: seg_imgs={seg_imgs_epoch}, cls_imgs={cls_imgs_epoch}")

        dist.barrier()

        # Validation on master only, same as original but TM transforms for spatial handling
        if is_master:
            torch.cuda.empty_cache()

            save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch_num}
            save_latest_path = os.path.join(snapshot_path, 'latest_tm_{}.pth'.format(epoch_num))
            if os.path.exists(os.path.join(snapshot_path, 'latest_tm_{}.pth'.format(epoch_num-1))):
                os.remove(os.path.join(snapshot_path, 'latest_tm_{}.pth'.format(epoch_num-1)))
                if os.path.islink(os.path.join(snapshot_path, 'latest_tm.pth')):
                    os.remove(os.path.join(snapshot_path, 'latest_tm.pth'))
            torch.save(save_dict, save_latest_path)
            os.system('ln -s ' + os.path.abspath(save_latest_path) + ' ' + os.path.join(snapshot_path, 'latest_tm.pth'))

            # swap to EMA weights if enabled
            swapped_to_ema = False
            if use_ema and (ema_state is not None):
                orig_state = copy.deepcopy(model.module.state_dict())
                model.module.load_state_dict(ema_state, strict=False)
                swapped_to_ema = True
            model.eval()
            total_performance = 0.0

            seg_val_set = [
                "BUS-BRA", "BUSIS", "BUSI", "CAMUS", "DDTI", "Fetal_HC", "KidneyUS",
                "private_Thyroid", "private_Kidney", "private_Fetal_Head", "private_Cardiac",
                "private_Breast_luminal", "private_Breast",
            ]
            seg_avg_performance = 0.0
            seg_val_loss_sum = 0.0
            seg_val_loss_count = 0

            for dataset_name in seg_val_set:
                num_classes = 2
                db_val = USdatasetSeg(
                    base_dir=os.path.join(args.root_path, "segmentation", dataset_name),
                    split="val",
                    list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
                    transform=ResizePadTM(output_size=[args.img_size, args.img_size]),
                    prompt=args.prompt,
                )
                val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=16)
                logging.info(f"[DBG] SEG VAL {dataset_name}: dataset_len={len(db_val)}, iters={len(val_loader)}")

                # Accumulate per-class dice only over valid cases
                sum_dice = np.zeros((num_classes-1,), dtype=np.float64)
                valid_count = np.zeros((num_classes-1,), dtype=np.float64)
                processed = 0
                preview_logged = False
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), disable=False):
                    bsz = sampled_batch['image'].size(0)
                    processed += bsz
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    if args.prompt and 'position_prompt' in sampled_batch and 'task_prompt' in sampled_batch and 'type_prompt' in sampled_batch and 'nature_prompt' in sampled_batch:
                        position_prompt = _to_bdim(sampled_batch['position_prompt'], 8, bsz).float()
                        task_prompt = _to_bdim(sampled_batch['task_prompt'], 2, bsz).float()
                        type_prompt = _to_bdim(sampled_batch['type_prompt'], 3, bsz).float()
                        nature_prompt = _to_bdim(sampled_batch['nature_prompt'], 2, bsz).float()
                        metric_i = omni_seg_test(
                            image, label, model, classes=num_classes, prompt=True,
                            type_prompt=type_prompt, nature_prompt=nature_prompt,
                            position_prompt=position_prompt, task_prompt=task_prompt
                        )
                    else:
                        metric_i = omni_seg_test(image, label, model, classes=num_classes)

                    # Compute validation loss (seg) with current batch
                    with torch.no_grad():
                        if args.prompt and 'position_prompt' in sampled_batch and 'task_prompt' in sampled_batch and 'type_prompt' in sampled_batch and 'nature_prompt' in sampled_batch:
                            outputs_val = model((image.cuda(), position_prompt.cuda(), task_prompt.cuda(), type_prompt.cuda(), nature_prompt.cuda()))[0]
                        else:
                            # Temporarily disable prompt flags for non-prompt forward
                            is_ddp = hasattr(model, 'module')
                            mref = model.module if is_ddp else model
                            restore_prompt = None
                            restore_swin_prompt = None
                            try:
                                if hasattr(mref, 'prompt') and getattr(mref, 'prompt', False):
                                    restore_prompt = mref.prompt
                                    mref.prompt = False
                                if hasattr(mref, 'swin') and hasattr(mref.swin, 'prompt') and getattr(mref.swin, 'prompt', False):
                                    restore_swin_prompt = mref.swin.prompt
                                    mref.swin.prompt = False
                                outputs_val = model(image.cuda())[0]
                            finally:
                                if restore_prompt is not None:
                                    mref.prompt = restore_prompt
                                if restore_swin_prompt is not None:
                                    mref.swin.prompt = restore_swin_prompt
                        # Ensure labels are 0/1 float for seg loss
                        lbl = label.cuda().float()
                        if lbl.max() > 1.0:
                            lbl = (lbl > 0).float()
                        l = seg_crit(outputs_val, lbl)
                        seg_val_loss_sum += l.item() * bsz
                        seg_val_loss_count += bsz
                    # metric_i: list of (dice, valid_flag) for classes 1..C-1
                    for ci in range(num_classes-1):
                        dice_val, is_valid = metric_i[ci]
                        if is_valid:
                            sum_dice[ci] += float(dice_val)
                            valid_count[ci] += 1.0

                    # Log one preview image per dataset
                    if use_wandb and (not preview_logged):
                        # read original image and mask, build panel
                        case_name = sampled_batch['case_name'][0]
                        img_path = os.path.join(args.root_path, 'segmentation', dataset_name, 'imgs', case_name)
                        gt_path = os.path.join(args.root_path, 'segmentation', dataset_name, 'masks', case_name)
                        orig = cv2.imread(img_path)
                        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                        orig_h, orig_w = orig.shape[:2]
                        # prediction for first item
                        with torch.no_grad():
                            if args.prompt and 'position_prompt' in sampled_batch and 'task_prompt' in sampled_batch and 'type_prompt' in sampled_batch and 'nature_prompt' in sampled_batch:
                                out_one = model((image[0:1].cuda(), position_prompt[0:1].cuda(), task_prompt[0:1].cuda(), type_prompt[0:1].cuda(), nature_prompt[0:1].cuda()))[0]
                            else:
                                is_ddp = hasattr(model, 'module')
                                mref = model.module if is_ddp else model
                                restore_prompt = None
                                restore_swin_prompt = None
                                try:
                                    if hasattr(mref, 'prompt') and getattr(mref, 'prompt', False):
                                        restore_prompt = mref.prompt
                                        mref.prompt = False
                                    if hasattr(mref, 'swin') and hasattr(mref.swin, 'prompt') and getattr(mref.swin, 'prompt', False):
                                        restore_swin_prompt = mref.swin.prompt
                                        mref.swin.prompt = False
                                    out_one = model(image[0:1].cuda())[0]
                                finally:
                                    if restore_prompt is not None:
                                        mref.prompt = restore_prompt
                                    if restore_swin_prompt is not None:
                                        mref.swin.prompt = restore_swin_prompt
                        pred = torch.argmax(torch.softmax(out_one, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                        pred_restored = _restore_mask_from_padded(pred, orig_h, orig_w, args.img_size, args.img_size)
                        pred_vis = (pred_restored > 0).astype(np.uint8) * 255
                        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        gt_vis = (gt > 0).astype(np.uint8) * 255
                        # stack images horizontally
                        pred_rgb = np.stack([pred_vis]*3, axis=-1)
                        gt_rgb = np.stack([gt_vis]*3, axis=-1)
                        panel = np.concatenate([orig, pred_rgb, gt_rgb], axis=1)
                        wandb.log({f'val/seg/{dataset_name}/preview': wandb.Image(panel, caption='origin | predict | gt')}, step=global_iter_num)
                        preview_logged = True

                logging.info(f"[DBG] SEG VAL {dataset_name}: processed_imgs={processed}")
                denom = np.maximum(valid_count, 1e-6)
                per_class_dice = sum_dice / denom
                performance = float(np.mean(per_class_dice))
                writer.add_scalar('info/val_seg_metric_tm_{}'.format(dataset_name), performance, epoch_num)
                seg_avg_performance += performance
                if use_wandb:
                    wandb.log({f'val/seg/{dataset_name}/dice': float(performance)}, step=global_iter_num)

            seg_avg_performance = seg_avg_performance / (len(seg_val_set)+1e-6)
            total_performance += seg_avg_performance
            writer.add_scalar('info/val_metric_seg_Total_tm', seg_avg_performance, epoch_num)
            if use_wandb:
                wandb.log({'val/seg/mean_dice': float(seg_avg_performance)}, step=global_iter_num)

            seg_val_loss = (seg_val_loss_sum / max(1, seg_val_loss_count)) if seg_val_loss_count > 0 else 0.0
            if use_wandb:
                wandb.log({'val/seg_loss': float(seg_val_loss)}, step=global_iter_num)

            cls_val_set = [
                "Appendix", "BUS-BRA", "BUSI", "Fatty-Liver", "private_Liver",
                "private_Breast_luminal", "private_Breast", "private_Appendix",
            ]
            cls_avg_performance = 0.0
            cls_val_loss_sum = 0.0
            cls_val_loss_count = 0

            for dataset_name in cls_val_set:
                num_classes = 4 if dataset_name == "private_Breast_luminal" else 2
                db_val = USdatasetCls(
                    base_dir=os.path.join(args.root_path, "classification", dataset_name),
                    split="val",
                    list_dir=os.path.join(args.root_path, "classification", dataset_name),
                    transform=ResizePadTM(output_size=[args.img_size, args.img_size]),
                    prompt=args.prompt,
                )
                val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=16)
                logging.info(f"[DBG] CLS VAL {dataset_name}: dataset_len={len(db_val)}, iters={len(val_loader)}")

                model.eval()
                label_list = []
                prediction_prob_list = []
                processed = 0
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), disable=False):
                    bsz = sampled_batch['image'].size(0)
                    processed += bsz
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    if args.prompt and 'position_prompt' in sampled_batch and 'task_prompt' in sampled_batch and 'type_prompt' in sampled_batch and 'nature_prompt' in sampled_batch:
                        position_prompt = _to_bdim(sampled_batch['position_prompt'], 8, bsz).float()
                        task_prompt = _to_bdim(sampled_batch['task_prompt'], 2, bsz).float()
                        type_prompt = _to_bdim(sampled_batch['type_prompt'], 3, bsz).float()
                        nature_prompt = _to_bdim(sampled_batch['nature_prompt'], 2, bsz).float()
                        with torch.no_grad():
                            output = model((image.cuda(), position_prompt.cuda(), task_prompt.cuda(), type_prompt.cuda(), nature_prompt.cuda()))
                    else:
                        with torch.no_grad():
                            is_ddp = hasattr(model, 'module')
                            mref = model.module if is_ddp else model
                            restore_prompt = None
                            restore_swin_prompt = None
                            try:
                                if hasattr(mref, 'prompt') and getattr(mref, 'prompt', False):
                                    restore_prompt = mref.prompt
                                    mref.prompt = False
                                if hasattr(mref, 'swin') and hasattr(mref.swin, 'prompt') and getattr(mref.swin, 'prompt', False):
                                    restore_swin_prompt = mref.swin.prompt
                                    mref.swin.prompt = False
                                output = model(image.cuda())
                            finally:
                                if restore_prompt is not None:
                                    mref.prompt = restore_prompt
                                if restore_swin_prompt is not None:
                                    mref.swin.prompt = restore_swin_prompt

                    logits = output[2] if num_classes == 4 else output[1]
                    # accumulate classification val loss
                    with torch.no_grad():
                        ce_val = (cls_crit_4 if num_classes == 4 else cls_crit_2)(logits, label.long().cuda())
                        cls_val_loss_sum += ce_val.item() * bsz
                        cls_val_loss_count += bsz
                    output_prob = torch.softmax(logits, dim=1).data.cpu().numpy()
                    label_list.append(label.numpy())
                    prediction_prob_list.append(output_prob)

                logging.info(f"[DBG] CLS VAL {dataset_name}: processed_imgs={processed}")
                label_list = np.expand_dims(np.concatenate((np.array(label_list[:-1]).flatten(), np.array(label_list[-1]).flatten())), axis=1).astype('uint8')
                label_list_OneHot = np.eye(num_classes)[label_list].squeeze(1)
                prediction_probs_reshaped = np.array(prediction_prob_list[:-1]).reshape(-1, num_classes)
                all_prediction_probs = np.concatenate((prediction_probs_reshaped, prediction_prob_list[-1]))
                performance = roc_auc_score(label_list_OneHot, all_prediction_probs, multi_class='ovo')

                writer.add_scalar('info/val_cls_metric_tm_{}'.format(dataset_name), performance, epoch_num)
                cls_avg_performance += performance

            cls_avg_performance = cls_avg_performance / (len(cls_val_set)+1e-6)
            total_performance += cls_avg_performance
            writer.add_scalar('info/val_metric_cls_Total_tm', cls_avg_performance, epoch_num)
            if use_wandb:
                wandb.log({'val/cls/mean_auc': float(cls_avg_performance)}, step=global_iter_num)

            cls_val_loss = (cls_val_loss_sum / max(1, cls_val_loss_count)) if cls_val_loss_count > 0 else 0.0
            if use_wandb:
                wandb.log({'val/cls_loss': float(cls_val_loss)}, step=global_iter_num)

            TotalAvgPerformance = total_performance / 2
            logging.info('This epoch %d Validation performance (TM): %f' % (epoch_num, TotalAvgPerformance))
            logging.info(f"[VAL] epoch={epoch_num} seg_mean={seg_avg_performance:.4f} cls_mean={cls_avg_performance:.4f} total_mean={TotalAvgPerformance:.4f} seg_loss={seg_val_loss:.4f} cls_loss={cls_val_loss:.4f}")
            writer.add_scalar('info/val_metric_TotalMean_tm', TotalAvgPerformance, epoch_num)
            if use_wandb:
                wandb.log({'val/total_mean': float(TotalAvgPerformance)}, step=global_iter_num)

            # Best snapshot mgmt (TM)
            best_flag_path = os.path.join(snapshot_path, 'best_model_tm.pth')
            # Track best via file name with score
            if (epoch_num == 0) or (TotalAvgPerformance >= omni_train_tm.best_performance if hasattr(omni_train_tm, 'best_performance') else -1):
                prev_epoch = getattr(omni_train_tm, 'best_epoch', None)
                prev_perf = getattr(omni_train_tm, 'best_performance', None)
                if prev_epoch is not None:
                    prev_path = os.path.join(snapshot_path, 'best_model_tm_{}_{}.pth'.format(prev_epoch, round(prev_perf, 4)))
                    if os.path.exists(prev_path):
                        os.remove(prev_path)
                    if os.path.islink(best_flag_path):
                        os.remove(best_flag_path)
                omni_train_tm.best_epoch = epoch_num
                omni_train_tm.best_performance = TotalAvgPerformance
                save_model_path = os.path.join(snapshot_path, 'best_model_tm_{}_{}.pth'.format(epoch_num, round(TotalAvgPerformance, 4)))
                torch.save(model.state_dict(), save_model_path)
                os.system('ln -s ' + os.path.abspath(save_model_path) + ' ' + best_flag_path)
                logging.info("[TM] save best model to {}".format(save_model_path))

            # restore original weights after EMA eval
            if swapped_to_ema:
                model.module.load_state_dict(orig_state, strict=False)
            model.train()

            # Compute validation summary and apply plateau LR if needed
            if seg_val_loss_count > 0:
                seg_val_loss_avg = seg_val_loss_sum / float(seg_val_loss_count)
            else:
                seg_val_loss_avg = float('inf')
            writer.add_scalar('val/seg_loss_avg', seg_val_loss_avg, epoch_num)
            logging.info(f"[VAL] epoch={epoch_num} seg_loss_avg={seg_val_loss_avg:.4f}")

            improved = seg_val_loss_avg < (best_val - 1e-6)
            if improved:
                best_val = seg_val_loss_avg
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= int(getattr(args, 'plateau_patience', 20)):
                # Decay LR by factor 0.5
                current_base_lr = max(1e-8, current_base_lr * 0.5)
                for param_group in optimizer.param_groups:
                    mult = (param_group.get('initial_lr', base_lr) / base_lr)
                    param_group['lr'] = current_base_lr * mult
                logging.info(f"[LR] Plateau detected. Reducing base LR to {current_base_lr:.6f}")
                writer.add_scalar('lr/base', current_base_lr, epoch_num)
                no_improve_epochs = 0

            # Early stopping check (on master)
            if args.early_stop_metric == 'total_mean':
                current_score = float(TotalAvgPerformance)
            elif args.early_stop_metric == 'seg_mean':
                current_score = float(seg_avg_performance)
            else:
                current_score = float(cls_avg_performance)

            if current_score > best_score + 1e-8:
                best_score = current_score
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                logging.info(f"[ES] No improvement epochs: {no_improve_epochs}/{args.early_stop_patience} (best={best_score:.5f}, current={current_score:.5f})")
                if use_wandb:
                    wandb.log({'early_stop/no_improve_epochs': no_improve_epochs, 'early_stop/best_score': best_score, 'early_stop/current_score': current_score}, step=global_iter_num)
                if no_improve_epochs >= args.early_stop_patience:
                    logging.info(f"[ES] Early stopping triggered at epoch {epoch_num}.")
                    if use_wandb:
                        wandb.summary['early_stopped_at'] = epoch_num
                        wandb.summary['best_score'] = best_score
                    # break outer loop by setting a flag
                    stop_training = True
                else:
                    stop_training = False
        
        # Broadcast stop flag to all ranks
        stop_tensor = torch.tensor([1 if (is_master and 'stop_training' in locals() and stop_training) else 0], device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            if is_master:
                logging.info("[ES] Stopping all ranks.")
            break

        model.train()

    writer.close()
    if is_master and use_wandb:
        wandb.finish()
    return "Training Finished (TM)!"