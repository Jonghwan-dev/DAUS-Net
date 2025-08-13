import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.transunet import TransUnetTM
from trainer import omni_train_tm
from configs.config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/', help='root dir for data')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch_size per gpu')
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml",
                    metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into non-overlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument('--pretrain_ckpt', type=str, help='pretrained checkpoint')

parser.add_argument('--prompt', action='store_true', help='using prompt for training')
parser.add_argument('--adapter_ft', action='store_true', help='using adapter for fine-tuning')
parser.add_argument('--del_outlier', action='store_true',
                    help='remove classification outliers listed in datasets/cls_outliers_organ.csv')

# Weights & Biases logging
parser.add_argument('--wandb_project', type=str, default='uusic25_tm', help='wandb project name')
parser.add_argument('--wandb_run', type=str, default=None, help='wandb run name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/user or team')
parser.add_argument('--wandb_off', action='store_true', help='disable wandb logging')

# Training control
parser.add_argument('--max_epochs', type=int, default=200, help='maximum training epochs')
parser.add_argument('--early_stop_patience', type=int, default=20, help='early stopping patience (epochs)')
parser.add_argument('--early_stop_metric', type=str, default='total_mean', choices=['total_mean', 'seg_mean', 'cls_mean'], help='metric to monitor for early stopping')
parser.add_argument('--plateau_patience', type=int, default=20, help='epochs without improvement before LR plateau decay')

# Optimization/scheduler options (TM)
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs for lr')
parser.add_argument('--min_lr_ratio', type=float, default=0.05, help='min lr = base_lr * min_lr_ratio (cosine)')
parser.add_argument('--head_lr_mult', type=float, default=1.5, help='lr multiplier for heads/prompt params')
parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='gradient clipping max norm (0 to disable)')
parser.add_argument('--amp', type=int, default=1, help='use AMP mixed precision (1 to enable, 0 to disable)')
parser.add_argument('--w_align', type=float, default=0.1, help='weight for prompt-pooled alignment loss')
parser.add_argument('--max_lora_scale', type=float, default=1.0, help='cap for prompt-controlled LoRA runtime scale')
parser.add_argument('--scale_mode', type=str, default='sigmoid', choices=['sigmoid','softplus','tanh'], help='activation for LoRA scale head')

# Classification loss options
parser.add_argument('--cls_loss', type=str, default='ce', choices=['ce', 'focal', 'ls'], help='classification loss type: cross-entropy, focal, or label smoothing')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing value if cls_loss=ls')

# Classification loss ramp-up between tasks
parser.add_argument('--w_cls_start', type=float, default=0.4, help='initial weight for classification loss')
parser.add_argument('--w_cls_end', type=float, default=1.0, help='final weight for classification loss after ramp epochs')
parser.add_argument('--w_cls_ramp_epochs', type=int, default=10, help='epochs to ramp classification loss weight')

# Prompt-conditioned classification controls
parser.add_argument('--film_scale', type=float, default=0.5, help='strength of FiLM modulation from prompt embedding for classifier')
parser.add_argument('--prior_lambda', type=float, default=0.3, help='weight to combine prompt-prior logits with image logits for classifier')

# EMA
parser.add_argument('--use_ema', action='store_true', help='use EMA weights for evaluation')
parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay')

# (MixUp/CutMix removed)

# Classification hard example mining / weighting
parser.add_argument('--cls_hard_weight_gamma', type=float, default=2.0, help='gamma for difficulty-based weighting of CE per-sample')
parser.add_argument('--cls_ohem_frac', type=float, default=0.0, help='fraction of hardest samples to keep per-batch (0 disables)')

# Segmentation-specific options
parser.add_argument('--seg_loss', type=str, default='bce_dice',
                    choices=['bce_dice', 'lovasz', 'focal_lovasz', 'focal_tversky'],
                    help='segmentation loss: BCE+Dice | Lovasz+Dice | 0.4*Focal+0.6*LovaszHinge | FocalTversky')
parser.add_argument('--seg_focal_gamma', type=float, default=2.0, help='focal gamma for focal-based seg losses')
parser.add_argument('--seg_focal_alpha_fg', type=float, default=0.8, help='foreground alpha for focal CE (0..1)')
parser.add_argument('--seg_tversky_alpha', type=float, default=0.5, help='alpha (FP weight) for Focal Tversky')
parser.add_argument('--seg_tversky_beta', type=float, default=0.5, help='beta (FN weight) for Focal Tversky')
parser.add_argument('--seg_tversky_gamma', type=float, default=1.0, help='gamma for Focal Tversky')
parser.add_argument('--seg_skip_bg_only_prob', type=float, default=0.0, help='probability to skip bg-only seg batches')
parser.add_argument('--seg_bce_weight', type=float, default=0.5, help='weight of BCEWithLogits in seg loss')
parser.add_argument('--seg_dice_weight', type=float, default=0.5, help='weight of Dice in seg loss')
parser.add_argument('--seg_auto_pos_weight', type=int, default=1, help='auto positive class weighting for BCE (1 to enable, 0 to disable)')
parser.add_argument('--seg_area_weight', type=float, default=0.1, help='weight for area consistency prior between pred and GT')
parser.add_argument('--seg_fp_weight', type=float, default=0.2, help='weight for hard false-positive top-k penalty')
parser.add_argument('--seg_fp_topk', type=float, default=0.1, help='fraction of background pixels to penalize as top-k')

# LoRA options for TransUNet encoder
parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (0 disables LoRA)')
parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA scaling alpha')
parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout probability')
parser.add_argument('--lora_only', action='store_true', help='train only LoRA adapters (freeze base linear weights)')

# Classification head variants
parser.add_argument('--cls_head_variant', type=str, default='linear',
                    choices=['linear', 'shared_mlp', 'per_head_mlp'],
                    help='classification head type: linear | shared_mlp | per_head_mlp')
parser.add_argument('--cls_dropout', type=float, default=0.3,
                    help='dropout for classification MLP trunks (if used)')


args = parser.parse_args()

config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Strengthen determinism across libraries
    try:
        os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
        # For deterministic cuBLAS (PyTorch recommends these for some ops)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(args.deterministic == 1)
    except Exception:
        pass

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Build TransUnetTM model (seg_out_ch=2 for binary seg by default)
    net = TransUnetTM(
        img_size=args.img_size,
        in_chans=1,
        seg_out_ch=2,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_only=args.lora_only,
        max_lora_scale=args.max_lora_scale,
        scale_mode=args.scale_mode,
        film_scale=args.film_scale,
        prior_lambda=args.prior_lambda,
        cls_head_variant=args.cls_head_variant,
        cls_dropout=args.cls_dropout,
    ).cuda()

    # Ensure all modules (seg, cls, prompt controllers, backbone, LoRA) are trainable
    for name, param in net.named_parameters():
        param.requires_grad = True
    # Print brief trainable parameter summary
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in net.parameters())
    print(f"[trainable] params: {n_trainable}/{n_total} ({100.0*n_trainable/max(1,n_total):.2f}%)")

    omni_train_tm(args, net, args.output_dir)