import torch
import torch.nn.functional as F
import numpy as np

from .metrics_loss import dice_coeff_3d_volume, iou_score_3d_volume, hd95_score_3d_volume, gaussian_3d

def accumulate_case_buffers(batch, logits, case_buffers, gaussian_1d, device):
    B, T, _, H, W = batch["ct"].shape
    h, w = logits.shape[-2], logits.shape[-1]
    # h,w = H,W
    gt = batch["mask"].float()
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    gt_bt = gt.view(B * T, 1, H, W)
    gt_full = gt_bt.cpu().numpy()[:, 0]

    if gaussian_1d is None or gaussian_1d.shape[0] != T:
        gaussian_1d = gaussian_3d(T, device=logits.device)
    # probs = torch.sigmoid(logits).detach().cpu().numpy()

    logits_np = logits.detach().cpu().numpy()


    start_i_list = batch.get("start_i", [0] * B)
    if hasattr(start_i_list, "cpu"):
        start_i_list = start_i_list.cpu().tolist()
    else:
        start_i_list = list(start_i_list) if len(start_i_list) == B else [0] * B
    case_ids = batch.get("case_id", [])
    class_names = batch.get("class_name", ["unknown"] * B)
    num_valid_z_list = batch.get("num_valid_z", [T] * B)
    spacing_z_list = batch.get("spacing_z", [1.0] * B)
    if torch.is_tensor(spacing_z_list):
        if spacing_z_list.dim() == 0:
            spacing_z_list = [float(spacing_z_list.item())] * B
        else:
            spacing_z_list = [float(spacing_z_list[b].item()) for b in range(B)]
    elif isinstance(spacing_z_list, (list, tuple)):
        spacing_z_list = [float(x) for x in spacing_z_list]
    else:
        spacing_z_list = [float(spacing_z_list)] * B

    spacing = batch["spacing"]
    spacing_np = spacing.cpu().numpy() if hasattr(spacing, "cpu") else np.array(spacing)
    if spacing_np.ndim == 1:
        spacing_np = np.expand_dims(spacing_np, 0)

    for b in range(B):
        cid = case_ids[b]
        start_i = int(start_i_list[b]) if b < len(start_i_list) else 0
        L = int(num_valid_z_list[b]) if b < len(num_valid_z_list) else T
        sz = float(spacing_z_list[b]) if b < len(spacing_z_list) else 1.0
        xy = spacing_np[b] if b < len(spacing_np) else np.array([1.0, 1.0])
        if cid not in case_buffers:
            case_buffers[cid] = {
                "numer": np.zeros((L, h, w), dtype=np.float32),
                "denom": np.zeros((L, h, w), dtype=np.float32),
                "gt_vol": np.zeros((L, H, W), dtype=np.float32),
                "L": L, "h": h, "w": w, "H": H, "W": W,
                "spacing_xy": (float(xy[0]), float(xy[1])),
                "spacing_z": sz,
                "class_name": class_names[b] if b < len(class_names) else "unknown",
            }
        buf = case_buffers[cid]
        for t in range(T):
            z_idx = start_i + t
            if z_idx < buf["L"]:
                buf["numer"][z_idx] += gaussian_1d[t].item() * logits_np[b * T + t, 0]
                buf["denom"][z_idx] += gaussian_1d[t].item()
        end_z = min(start_i + T, buf["L"])
        n_slices = end_z - start_i
        buf["gt_vol"][start_i : end_z] = gt_full[b * T : b * T + n_slices]
    return gaussian_1d


def compute_3d_metrics_from_buffers(case_buffers, out_hw: int | None = None):
    liver_dice_3d,  liver_iou_3d,  liver_hd95_3d  = [], [], []
    lesion_dice_3d, lesion_iou_3d, lesion_hd95_3d = [], [], []
    tumor_dice_3d,  tumor_iou_3d,  tumor_hd95_3d  = [], [], []
    for cid, buf in case_buffers.items():
        denom = np.maximum(buf["denom"], 1e-8)

        fused_logit = (buf["numer"] / denom).astype(np.float32)
        gt_vol = buf["gt_vol"].astype(np.float32)

        target_h = out_hw if out_hw is not None else gt_vol.shape[1]
        target_w = out_hw if out_hw is not None else gt_vol.shape[2]

        logit_t = torch.from_numpy(fused_logit).float().unsqueeze(1)  # (L,1,h,w)

        if logit_t.shape[-2] != target_h or logit_t.shape[-1] != target_w:
            logit_t = F.interpolate(logit_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
            print("================================================")
            print("[DEBUG] logit_t shape mismatch: ", logit_t.shape, "->", target_h, target_w)
            print("================================================")

        fused_prob = torch.sigmoid(logit_t)[:, 0].cpu().numpy() 
        pred_vol = (fused_prob > 0.5).astype(np.float32)
        gt_vol = (gt_vol > 0.5).astype(np.float32)

        # fused_prob = (buf["numer"] / denom).astype(np.float32)
        # gt_vol = buf["gt_vol"].astype(np.float32)

        # # target -> out_hw(1), -> GT size(2)
        # target_h = out_hw if out_hw is not None else gt_vol.shape[1]
        # target_w = out_hw if out_hw is not None else gt_vol.shape[2]

        # # pred prob만 bilinear 업샘플
        # pred_t = torch.from_numpy(fused_prob).float().unsqueeze(1)  # (L,1,h,w)
        # pred_t = F.interpolate(pred_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        # fused_prob = pred_t[:, 0].cpu().numpy()  # (L,target_h,target_w)
        # pred_vol = (fused_prob > 0.5).astype(np.float32)
        # gt_vol = (gt_vol > 0.5).astype(np.float32)

        assert pred_vol.shape == gt_vol.shape, (
            f"[compute_3d_metrics_from_buffers] pred/gt shape mismatch for case={cid}: "
            f"pred={pred_vol.shape}, gt={gt_vol.shape}"
        )

        # sp = (buf["spacing_xy"][0], buf["spacing_xy"][1], buf["spacing_z"])
        sp = (buf["spacing_z"],buf["spacing_xy"][0], buf["spacing_xy"][1])

        dice_3d = dice_coeff_3d_volume(pred_vol, gt_vol)
        iou_3d = iou_score_3d_volume(pred_vol, gt_vol)
        hd95_3d = hd95_score_3d_volume(pred_vol, gt_vol, spacing_3d=sp)
        if buf["class_name"] == "liver":
            liver_dice_3d.append(dice_3d); liver_iou_3d.append(iou_3d); liver_hd95_3d.append(hd95_3d)
        elif buf["class_name"] == "tumor":                          
            tumor_dice_3d.append(dice_3d); tumor_iou_3d.append(iou_3d); tumor_hd95_3d.append(hd95_3d)
        else:  # lesion
            lesion_dice_3d.append(dice_3d); lesion_iou_3d.append(iou_3d); lesion_hd95_3d.append(hd95_3d)
    def _mean(x, d=0.0):
        return float(np.nanmean(x)) if x else d
    def _std(x, d=0.0):
        return float(np.nanstd(x)) if len(x) > 1 else d
    all_d = liver_dice_3d + lesion_dice_3d + tumor_dice_3d
    all_i = liver_iou_3d  + lesion_iou_3d  + tumor_iou_3d
    all_h = liver_hd95_3d + lesion_hd95_3d + tumor_hd95_3d
    return {
        "dice": _mean(all_d),
        "iou": _mean(all_i),
        "hd95": _mean(all_h),
        "dice_std": _std(all_d),
        "iou_std": _std(all_i),
        "hd95_std": _std(all_h),
        "dice_liver": _mean(liver_dice_3d),
        "iou_liver": _mean(liver_iou_3d),
        "hd95_liver": _mean(liver_hd95_3d),
        "dice_std_liver": _std(liver_dice_3d),
        "iou_std_liver": _std(liver_iou_3d),
        "hd95_std_liver": _std(liver_hd95_3d),
        "dice_lesion": _mean(lesion_dice_3d),
        "iou_lesion": _mean(lesion_iou_3d),
        "hd95_lesion": _mean(lesion_hd95_3d),
        "dice_std_lesion": _std(lesion_dice_3d),
        "iou_std_lesion": _std(lesion_iou_3d),
        "hd95_std_lesion": _std(lesion_hd95_3d),
        "num_samples_liver": len(liver_dice_3d),
        "num_samples_lesion": len(lesion_dice_3d),
        "dice_tumor":      _mean(tumor_dice_3d),
        "iou_tumor":       _mean(tumor_iou_3d),
        "hd95_tumor":      _mean(tumor_hd95_3d),
        "dice_std_tumor":  _std(tumor_dice_3d),
        "iou_std_tumor":   _std(tumor_iou_3d),
        "hd95_std_tumor":  _std(tumor_hd95_3d),
        "num_samples_tumor": len(tumor_dice_3d),
    }