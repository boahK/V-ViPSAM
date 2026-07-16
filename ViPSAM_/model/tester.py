import os
from collections import defaultdict
from typing import Collection, Dict, Optional, Set

import torch
import torch.nn.functional as F

from .metrics_loss import seg_loss
from .util import accumulate_case_buffers, compute_3d_metrics_from_buffers
from data import patient_key_from_case_id
from data.prepare_data import load_volume, nonempty_slices
from utils import generate_box_prompt, generate_click_prompt, save_epoch_visual, update_overlay_and_save


def validation_loop(
    model,
    val_loader,
    cfg,
    device,
    vis_dir=None,
    epoch=None,
    save_vis=True,
    dataset_name="smc",
    vis_data_list=None,
    vis_patient_keys=None,
    vis_max_patients=0,
):
    model.eval()
    va_loss = 0.0
    image_size = cfg.image_size
    prompt_type = cfg.prompt_type
    dataset_name = str(dataset_name).lower()

    case_buffers = {}
    gaussian_1d = None

    if dataset_name == "apis":
        required_classes = ("lesion",)
    elif dataset_name == "hecktor":
        required_classes = ("tumor",)
    else:                                    
        required_classes = ("liver", "lesion")
        
    allow_patients = None
    lesion_z_by_pk = {}
    if save_vis and vis_dir is not None and epoch is None:
        if vis_data_list is not None:
            allow_patients = _resolve_vis_patients(vis_data_list, vis_patient_keys, vis_max_patients)
            if dataset_name == "smc":
                lesion_z_by_pk = _lesion_native_z_by_patient(vis_data_list)

    saved_cases = {}
    
    ck = 0

    with torch.no_grad():
        overlay_cache = {}
        for batch_idx, batch in enumerate(val_loader):
            ct = batch["ct"].to(device)
            mr = batch["mr"].to(device)
            gt = batch["mask"].float().to(device)
            B, T, _, H, W = ct.shape
            if batch_idx == 0:
                print("[Val] ct", ct.shape, "gt", gt.shape)
            gt_bt = gt.unsqueeze(2).view(B * T, 1, H, W)

            zi = batch.get("z_indices")
            if zi is None:
                raise KeyError("batch must include z_indices from dataset (CTMRDataset / APISDataset).")

            start_i_list = batch.get("start_i", [0] * B)
            if hasattr(start_i_list, "cpu"):
                start_i_list = start_i_list.cpu().tolist()
            else:
                start_i_list = list(start_i_list) if len(start_i_list) == B else [0] * B

            case_ids = batch.get("case_id", [])
            class_names = batch.get("class_name", ["unknown"] * B)
            patient_keys = [patient_key_from_case_id(cid) for cid in case_ids]

            if prompt_type == "box":
                boxes = generate_box_prompt(gt_bt, bbox_shift=0, image_size=image_size, fixed_shift=5).to(device)
                points, point_labels = None, None
            else:
                points, point_labels = generate_click_prompt(gt_bt, pt_label=1, image_size=image_size)
                points = points.to(device)
                point_labels = point_labels.to(device)
                boxes = None

            logits, _ = model(ct, mr, boxes=boxes, points=points, point_labels=point_labels, masks=None)
            logits_up = F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=False)
            # logits_up = F.interpolate(logits, size=(image_size, image_size), mode="nearest")


            va_loss += seg_loss(logits_up, gt_bt, use_bce=cfg.use_bce).item()
            gaussian_1d = accumulate_case_buffers(batch, logits_up, case_buffers, gaussian_1d, device)

            if ck < 5:
                ck += 1
                print("===================== Val_logits_size", logits_up.shape, "Val_gt_size", gt_bt.shape)

            if not save_vis or vis_dir is None:
                continue

            mid = T // 2
            logits_vis = logits_up
            gt_vis = gt_bt

            if epoch is not None:
                if batch_idx == 0:
                    cn = class_names[0] if class_names else "unknown"
                    out_png = os.path.join(vis_dir, f"epoch_{epoch:03d}_{cn}.png")
                    save_epoch_visual(
                        ct[0, mid], mr[0, mid],
                        gt_vis[mid : mid + 1],
                        logits_vis[mid : mid + 1],
                        out_png=out_png,
                        image_size=image_size,
                        class_name=cn,
                    )
                continue

            zi = zi.to(device) if hasattr(zi, "to") else zi
            for b in range(B):
                if b >= len(patient_keys) or b >= len(class_names) or b >= len(start_i_list):
                    continue
                patient_key = patient_keys[b]
                class_name = class_names[b]
                case_id = case_ids[b]
                if allow_patients is not None and patient_key not in allow_patients:
                    continue
                for t in range(T):
                    z_native = int(zi[b, t].item())  # 실제 native z
                    # SMC: liver는 lesion 존재 z에서만 참여
                    if dataset_name == "smc" and class_name == "liver":
                        lz = lesion_z_by_pk.get(patient_key)
                        if lz is not None and z_native not in lz:
                            continue
                    key = (patient_key, class_name)
                    if key not in saved_cases:
                        saved_cases[key] = set()
                    if (case_id, z_native) in saved_cases[key]:
                        continue
                    saved_cases[key].add((case_id, z_native))
                    cache_key = (patient_key, z_native)
                    ct_vis = ct[b, t]
                    mr_vis = mr[b, t]
                    gt_vis_b = gt_vis[b * T + t].unsqueeze(0)
                    logits_vis_b = logits_vis[b * T + t].unsqueeze(0)
                    update_overlay_and_save(
                        overlay_cache, cache_key,
                        ct_vis, mr_vis, gt_vis_b, logits_vis_b,
                        class_name, vis_dir, image_size,
                        required_classes=required_classes,
                    )

    num_batches = max(1, len(val_loader))
    va_loss /= num_batches
    metrics = compute_3d_metrics_from_buffers(case_buffers)
    return {"loss": va_loss, **metrics, "per_sample_metrics": []}


def _resolve_vis_patients(
    data_list,
    vis_patient_keys: Optional[Collection[str]],
    vis_max_patients: int,
) -> Optional[Set[str]]:
    if vis_patient_keys is not None:
        s = {str(x).strip() for x in vis_patient_keys if str(x).strip()}
        return s if s else None
    if vis_max_patients and vis_max_patients > 0:
        seen = []
        for c in data_list:
            pk = patient_key_from_case_id(c["case_id"])
            if pk not in seen:
                seen.append(pk)
            if len(seen) >= vis_max_patients:
                break
        return set(seen) if seen else None
    return None


def _lesion_native_z_by_patient(data_list) -> Dict[str, Set[int]]:
    by_pk: Dict[str, Set[int]] = defaultdict(set)
    for c in data_list:
        if c.get("class_name") != "lesion":
            continue
        pk = patient_key_from_case_id(c["case_id"])
        m = load_volume(c["label"])
        by_pk[pk].update(int(x) for x in nonempty_slices(m).tolist())
    return dict(by_pk)