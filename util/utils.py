import numpy as np
try:
    from medpy import metric as _medpy_metric
except Exception:
    _medpy_metric = None
import torch
import torch.nn as nn
import cv2
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class LovaszHingeLoss(nn.Module):
    """Binary Lovasz hinge loss wrapper.

    Expects logits of shape [N, 1, H, W] or [N, H, W] and binary targets [N, 1, H, W] or [N, H, W].
    Uses per-image reduction as commonly recommended for segmentation.
    """
    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if 'lovasz_hinge' not in globals() or lovasz_hinge is None:
            raise RuntimeError(
                "lovasz_hinge not found. Please install LovaszSoftmax (submodule) or ensure import path is correct.")
        # Squeeze potential channel dim if present
        if inputs.dim() == 4 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        return lovasz_hinge(inputs, targets, per_image=self.per_image)


class DiceLovaszLoss(nn.Module):
    """Combined Dice + Lovasz hinge loss for binary segmentation.

    - Dice is computed via existing DiceLoss (with softmax over 2 classes).
    - Lovasz hinge is computed on the foreground logit channel.
    """
    def __init__(self, lovasz_weight: float = 0.6, dice_weight: float = 0.4, n_classes: int = 2):
        super().__init__()
        assert n_classes == 2, "DiceLovaszLoss currently supports binary (2-class) segmentation."
        self.lovasz_weight = float(lovasz_weight)
        self.dice_weight = float(dice_weight)
        self._dice = DiceLoss(n_classes=n_classes)
        self._lovasz = LovaszHingeLoss(per_image=True)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Dice over softmax probs of 2 classes
        dice = self._dice(inputs, targets, softmax=True)
        # Lovasz hinge over foreground logit (channel 1)
        if inputs.dim() == 4 and inputs.size(1) >= 2:
            fg_logit = inputs[:, 1:2, ...]
        else:
            raise ValueError("Expected inputs shape [N, 2, H, W] for binary segmentation logits.")
        fg_target = (targets > 0).float()
        lovasz = self._lovasz(fg_logit, fg_target)
        return self.lovasz_weight * lovasz + self.dice_weight * dice


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional alpha class weighting.

    Args:
        gamma: focusing parameter (>=0).
        alpha: None or Tensor/list of shape [C] with class weights.
        reduction: 'mean' | 'sum' | 'none'
    Expect logits of shape [N, C] and targets of shape [N].
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha if alpha is not None else None)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy per-sample
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # pt = softmax prob of true class
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal


class FocalLovaszHingeLoss(nn.Module):
    """0.4·Focal + 0.6·Lovasz-Hinge for binary segmentation.

    - Supports logits with shape [N, 2, H, W] (preferred) or [N, 1, H, W].
    - Focal is computed as per-pixel cross-entropy focal with class weights (alpha).
      If `logits` has 1 channel, we approximate CE by stacking [neg, pos] logits.
    - Lovasz hinge is applied to the foreground logit.

    Args:
        focal_weight: weight for focal term (default 0.4)
        lovasz_weight: weight for lovasz term (default 0.6)
        gamma: focal focusing parameter
        alpha_fg: foreground class weight (alpha for CE over 2 classes). Background alpha becomes (1 - alpha_fg).
        per_image: lovasz per-image reduction
    """
    def __init__(self,
                 focal_weight: float = 0.4,
                 lovasz_weight: float = 0.6,
                 gamma: float = 2.0,
                 alpha_fg: float = 0.75,
                 per_image: bool = True):
        super().__init__()
        self.focal_w = float(focal_weight)
        self.lovasz_w = float(lovasz_weight)
        self.gamma = float(gamma)
        self.alpha_fg = float(alpha_fg)
        self._lovasz = LovaszHingeLoss(per_image=per_image)

    @staticmethod
    def _to_two_channel_logits(logits: torch.Tensor) -> torch.Tensor:
        # Convert [N,1,H,W] to pseudo two-channel by stacking [-x, x]
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [N,C,H,W], got {tuple(logits.shape)}")
        if logits.size(1) == 2:
            return logits
        if logits.size(1) == 1:
            x = logits
            return torch.cat([-x, x], dim=1)
        # If C>2, take the last (foreground) vs background (assume channel 0)
        fg = logits[:, 1:2, ...]
        bg = logits[:, 0:1, ...]
        return torch.cat([bg, fg], dim=1)

    def _pixelwise_focal_ce(self, logits_2ch: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel focal CE over 2-class logits.
        logits_2ch: [N,2,H,W]; target_bin: [N,H,W] in {0,1}
        """
        N, C, H, W = logits_2ch.shape
        assert C == 2
        # Flatten to [N*H*W, C] and [N*H*W]
        logits_flat = logits_2ch.permute(0, 2, 3, 1).reshape(-1, 2)
        target_flat = target_bin.reshape(-1).long()
        # Class weights: [bg, fg]
        alpha = torch.tensor([1.0 - self.alpha_fg, self.alpha_fg], dtype=logits_flat.dtype, device=logits_flat.device)
        ce = torch.nn.functional.cross_entropy(logits_flat, target_flat, weight=alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Prepare target as binary [N,H,W]
        if target.dim() == 4 and target.size(1) == 1:
            tgt = target[:, 0, ...]
        elif target.dim() == 3:
            tgt = target
        else:
            raise ValueError(f"Expected target [N,H,W] or [N,1,H,W], got {tuple(target.shape)}")
        tgt = (tgt > 0).to(dtype=logits.dtype)

        # Focal over 2-class logits
        logits_2ch = self._to_two_channel_logits(logits)
        focal = self._pixelwise_focal_ce(logits_2ch, tgt)

        # Lovasz hinge on foreground logit
        fg_logit = logits_2ch[:, 1:2, ...]
        lovasz = self._lovasz(fg_logit, tgt)

        return self.focal_w * focal + self.lovasz_w * lovasz


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for binary segmentation.

    Tversky = TP / (TP + alpha * FP + beta * FN)
    Focal Tversky Loss = (1 - Tversky) ** gamma

    Supports logits [N, 1 or 2, H, W] and binary targets [N, H, W] or [N,1,H,W].

    Args:
        alpha: weight for FP
        beta: weight for FN
        gamma: focusing parameter (>0), commonly 0.75 or 1.0
        smooth: small constant to avoid div-by-zero
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    @staticmethod
    def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [N,C,H,W], got {tuple(logits.shape)}")
        if logits.size(1) == 1:
            return torch.sigmoid(logits[:, 0, ...])  # [N,H,W]
        # Use softmax foreground channel 1
        prob = torch.softmax(logits, dim=1)
        return prob[:, 1, ...]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Prepare binary target [N,H,W]
        if target.dim() == 4 and target.size(1) == 1:
            tgt = target[:, 0, ...]
        elif target.dim() == 3:
            tgt = target
        else:
            raise ValueError(f"Expected target [N,H,W] or [N,1,H,W], got {tuple(target.shape)}")
        tgt = (tgt > 0).to(dtype=logits.dtype)

        p = self._fg_prob(logits)
        # Compute TP, FP, FN
        tp = (p * tgt).sum(dim=(1, 2))
        fp = (p * (1.0 - tgt)).sum(dim=(1, 2))
        fn = ((1.0 - p) * tgt).sum(dim=(1, 2))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = (1.0 - tversky) ** self.gamma
        return loss.mean()

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    ps = pred.sum()
    gs = gt.sum()
    if ps > 0 and gs > 0:
        if _medpy_metric is not None:
            dice = _medpy_metric.binary.dc(pred, gt)
        else:
            inter = (pred & gt).sum()
            dice = (2.0 * inter) / (ps + gs + 1e-6)
        return dice, True
    elif ps > 0 and gs == 0:
        return 0, False
    elif ps == 0 and gs > 0:
        return 0, True
    else:
        return 0, False


def omni_seg_test(image, label, net, classes, ClassStartIndex=1, test_save_path=None, case=None,
                  prompt=False,
                  type_prompt=None,
                  nature_prompt=None,
                  position_prompt=None,
                  task_prompt=None
                  ):
    label = label.squeeze(0).cpu().detach().numpy()
    image_save = image.squeeze(0).cpu().detach().numpy()
    input = image.cuda()
    if prompt:
        position_prompt = position_prompt.cuda()
        task_prompt = task_prompt.cuda()
        type_prompt = type_prompt.cuda()
        nature_prompt = nature_prompt.cuda()
    net.eval()
    with torch.no_grad():
        if prompt:
            seg_out = net((input, position_prompt, task_prompt, type_prompt, nature_prompt))[0]
        else:
            # If the wrapped model was built with prompt=True, temporarily disable it
            # to ensure non-prompt forward works correctly.
            is_ddp = hasattr(net, 'module')
            model_ref = net.module if is_ddp else net
            restore_prompt = None
            restore_swin_prompt = None
            try:
                if hasattr(model_ref, 'prompt') and getattr(model_ref, 'prompt', False):
                    restore_prompt = model_ref.prompt
                    model_ref.prompt = False
                if hasattr(model_ref, 'swin') and hasattr(model_ref.swin, 'prompt') and getattr(model_ref.swin, 'prompt', False):
                    restore_swin_prompt = model_ref.swin.prompt
                    model_ref.swin.prompt = False
                seg_out = net(input)[0]
            finally:
                if restore_prompt is not None:
                    model_ref.prompt = restore_prompt
                if restore_swin_prompt is not None:
                    model_ref.swin.prompt = restore_swin_prompt
        out_label_back_transform = torch.cat(
            [seg_out[:, 0:1], seg_out[:, ClassStartIndex:ClassStartIndex+classes-1]], axis=1)
        out = torch.argmax(torch.softmax(out_label_back_transform, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        if classes == 2:
            # Binary seg: be robust to mask values in {0,255} or any positive label.
            metric_list.append(calculate_metric_percase(prediction > 0, label > 0))
            break
        else:
            metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        image = (image_save - np.min(image_save)) / (np.max(image_save) - np.min(image_save))
        cv2.imwrite(test_save_path + '/'+case + "_pred.png", (prediction*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+case + "_img.png", ((image.squeeze(0))*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+case + "_gt.png", (label*255).astype(np.uint8))
    return metric_list


# --- Practical binary segmentation loss: BCEWithLogits + Dice ---
class BCEWithLogitsDiceLoss(nn.Module):
    """Combined BCEWithLogits + Dice loss for binary segmentation.

    Accepts logits of shape [N, C, H, W] where C==1 or C>=2.
    If C>=2 and use_fg_channel=True, uses foreground channel (index 1).
    Targets expected as [N, H, W] or [N, 1, H, W] with 0/1 values.

    Args:
        bce_weight: weight for BCE term.
        dice_weight: weight for Dice term.
        auto_pos_weight: if True, compute pos_weight per batch as neg/pos.
        pos_weight: optional fixed pos_weight tensor or float for BCE.
        use_fg_channel: when logits have 2 channels, use channel 1 as fg.
        eps: small epsilon for Dice stability.
    """
    def __init__(self,
                 bce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 auto_pos_weight: bool = True,
                 pos_weight=None,
                 use_fg_channel: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.bce_w = float(bce_weight)
        self.dice_w = float(dice_weight)
        self.auto_pos = bool(auto_pos_weight)
        self.register_buffer('pos_weight_buf', None)
        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.tensor(float(pos_weight), dtype=torch.float32)
            self.register_buffer('pos_weight_buf', pos_weight)
        self.use_fg = bool(use_fg_channel)
        self.eps = float(eps)

    def _select_fg_logit(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [N,C,H,W], got {tuple(logits.shape)}")
        if logits.size(1) == 1:
            return logits
        if logits.size(1) >= 2 and self.use_fg:
            return logits[:, 1:2, ...]
        # fallback: average channels
        return logits.mean(dim=1, keepdim=True)

    def _prepare_target(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0, ...]
        if target.dim() == 3:
            return target.float()
        raise ValueError(f"Expected target [N,H,W] or [N,1,H,W], got {tuple(target.shape)}")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fg_logit = self._select_fg_logit(logits)
        y = self._prepare_target(target)
        # BCE with logits
        pos_weight = None
        if self.auto_pos:
            num_pos = y.sum()
            num_total = torch.tensor(y.numel(), dtype=y.dtype, device=y.device)
            num_neg = num_total - num_pos
            # avoid div by zero
            pw = (num_neg + self.eps) / (num_pos + self.eps)
            pos_weight = pw.detach()
        elif self.pos_weight_buf is not None:
            pos_weight = self.pos_weight_buf
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            fg_logit.squeeze(1), y, pos_weight=pos_weight, reduction='mean')
        # Dice
        prob = torch.sigmoid(fg_logit)
        prob = prob.squeeze(1)
        intersect = (prob * y).sum()
        denom = prob.sum() + y.sum()
        dice = 1.0 - (2.0 * intersect + self.eps) / (denom + self.eps)
        return self.bce_w * bce + self.dice_w * dice
