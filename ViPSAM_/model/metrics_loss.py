import torch, torch.nn.functional as F
import numpy as np

from monai.losses import DiceCELoss
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, gaussian_filter

#-------------------------
# -------2D metrics-------
#-------------------------
def dice_coeff(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice

def iou_score(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou

def hd95_score(logits: torch.Tensor,
                       target: torch.Tensor,
                       spacing_b2: torch.Tensor,
                       connectivity: int = 1) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > 0.5)
    gt = (target > 0.5)

    B = pred.shape[0]
    struct = generate_binary_structure(2, connectivity)

    hd = np.empty(B, dtype=np.float32)

    spacing_np = spacing_b2.detach().cpu().numpy()  

    for b in range(B):
        P = pred[b, 0].detach().cpu().numpy()
        T = gt[b, 0].detach().cpu().numpy()
        sp = (float(spacing_np[b, 0]), float(spacing_np[b, 1]))

        p_any, t_any = P.any(), T.any()

        if (not p_any) and (not t_any):
            hd[b] = 0.0
            continue
        if (not p_any) or (not t_any):
            hd[b] = np.nan
            continue

        P_surf = P ^ binary_erosion(P, structure=struct, border_value=0)
        T_surf = T ^ binary_erosion(T, structure=struct, border_value=0)

        dt_T = distance_transform_edt(~T_surf, sampling=sp)
        dt_P = distance_transform_edt(~P_surf, sampling=sp)

        d1 = dt_T[P_surf]
        d2 = dt_P[T_surf]

        hd1 = np.percentile(d1, 95) if d1.size else 0.0
        hd2 = np.percentile(d2, 95) if d2.size else 0.0
        hd[b] = max(hd1, hd2)

    return torch.from_numpy(hd).to(device=logits.device, dtype=torch.float32)

#-------------------------
# -------3D metrics-------
#-------------------------
def gaussian_3d(T: int, sigma_scale: float = 1.0 / 8.0, device=None, dtype=torch.float32):
    tmp = np.zeros(T, dtype=np.float64)
    tmp[T // 2] = 1.0
    sigma = max(T * sigma_scale, 1e-3)
    g = gaussian_filter(tmp, sigma=sigma, mode="constant", cval=0.0)
    g = torch.from_numpy(g.astype(np.float32))
    if g.max() > 0:
        g = g / g.max()
    g = g.clamp(min=torch.finfo(torch.float32).eps)
    if device is not None:
        g = g.to(device=device, dtype=dtype)
    return g

def dice_coeff_3d_volume(pred_binary: np.ndarray, target_binary: np.ndarray, eps: float = 1e-6):
    p = (np.asarray(pred_binary, dtype=np.float64).ravel() > 0.5)
    t = (np.asarray(target_binary, dtype=np.float64).ravel()> 0.5)
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return float((2.0 * inter + eps) / (union + eps))

def iou_score_3d_volume(pred_binary: np.ndarray, target_binary: np.ndarray, eps: float = 1e-6):
    p = (np.asarray(pred_binary, dtype=np.float64).ravel() > 0.5)
    t = (np.asarray(target_binary, dtype=np.float64).ravel() > 0.5)
    inter = np.logical_and(p, t).sum()
    union = p.sum() + t.sum() - inter
    if union <= 0 and inter <= 0:
        return 1.0
    return float((inter + eps) / (union + eps))

def hd95_score_3d_volume(pred_binary: np.ndarray, target_binary: np.ndarray,
                         spacing_3d=(1.0, 1.0, 1.0), connectivity: int = 1):
    P = np.asarray(pred_binary, dtype=bool)
    T_gt = np.asarray(target_binary, dtype=bool)
    if P.shape != T_gt.shape:
        return np.nan
    if not P.any() and not T_gt.any():
        return 0.0
    if not P.any() or not T_gt.any():
        return np.nan
    struct = generate_binary_structure(3, connectivity)
    P_surf = np.logical_xor(P, binary_erosion(P, structure=struct, border_value=0))
    T_surf = np.logical_xor(T_gt, binary_erosion(T_gt, structure=struct, border_value=0))
    sp = (float(spacing_3d[0]), float(spacing_3d[1]), float(spacing_3d[2]))
    dt_T = distance_transform_edt(~T_surf, sampling=sp)
    dt_P = distance_transform_edt(~P_surf, sampling=sp)
    d1 = dt_T[P_surf]
    d2 = dt_P[T_surf]
    hd1 = np.percentile(d1, 95) if d1.size else 0.0
    hd2 = np.percentile(d2, 95) if d2.size else 0.0
    return float(max(hd1, hd2))

def soft_dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

# loss functions
def seg_loss(logits, target, use_bce=True, eps=1e-6):
    _dice_ce = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    dice_loss = soft_dice_loss(logits, target, eps=eps)
    if not use_bce:
        return dice_loss
    # bce = F.binary_cross_entropy_with_logits(logits, target)
    # return 0.7*dice_loss + 0.3*bce
    return _dice_ce(logits, target)

# def soft_tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1e-6):
#     p = torch.sigmoid(logits)
#     tp = (p * target).sum(dim=(1, 2, 3))
#     fp = (p * (1.0 - target)).sum(dim=(1, 2, 3))
#     fn = ((1.0 - p) * target).sum(dim=(1, 2, 3))
#     t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
#     return 1.0 - t.mean()

# def seg_loss(logits, target, use_bce=True, eps=1e-6,
#              tversky_alpha=0.3, tversky_beta=0.7, tversky_weight=0.5):
#     dice_loss = soft_dice_loss(logits, target, eps=eps)
#     if not use_bce:
#         return dice_loss
#     tv = soft_tversky_loss(logits, target, alpha=tversky_alpha, beta=tversky_beta, eps=eps)
#     return dice_loss + tversky_weight * tv

# def seg_loss(logits, target, use_bce=True, eps=1e-6, focal_gamma=2.0, alpha=0.25):
#     dice = soft_dice_loss(logits, target, eps=eps)

#     if not use_bce:
#         return dice

#     bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
#     prob = torch.sigmoid(logits)
#     p_t  = prob * target + (1 - prob) * (1 - target)
#     focal = bce * ((1 - p_t) ** focal_gamma)

#     # alpha -> foreground=alpha, background=1-alpha
#     alpha_t = alpha * target + (1 - alpha) * (1 - target)
#     focal   = (alpha_t * focal).mean()

#     return dice + focal
