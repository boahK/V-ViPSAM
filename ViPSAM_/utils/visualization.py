import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def _to_numpy_img(x):
    x = x.detach().cpu().clamp(0, 1).numpy()
    x = x.transpose(1, 2, 0)  
    return x

def _to_numpy_mask(x):
    if x.shape[1] == 1:
        x = torch.sigmoid(x)
    x = x.detach().cpu().squeeze().clamp(0, 1).numpy()  
    return x

def save_epoch_visual_overlay(ct, mr,gt_liver, gt_lesion, pred_liver, pred_lesion, out_png, image_size=512,class_labels=("Liver", "Lesion")):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    ct_np = _to_numpy_img(ct)
    mr_np = _to_numpy_img(mr)
    h, w = ct_np.shape[:2]

    gt_l = gt_liver.detach().cpu().squeeze().numpy().reshape(h, w)
    gt_le = gt_lesion.detach().cpu().squeeze().numpy().reshape(h, w)
    pr_l = np.squeeze(_to_numpy_mask(pred_liver)).reshape(h, w)
    pr_le = np.squeeze(_to_numpy_mask(pred_lesion)).reshape(h, w)

    liver_color = np.array([1, 0, 0, 0.6])
    lesion_color = np.array([0, 1, 0, 0.6])

    gt_overlay = np.zeros((h, w, 4), dtype=np.float32)
    gt_overlay[gt_l > 0.5] = liver_color
    gt_overlay[gt_le > 0.5] = lesion_color

    pred_overlay = np.zeros((h, w, 4), dtype=np.float32)
    pred_overlay[pr_l > 0.5] = liver_color
    pred_overlay[pr_le > 0.5] = lesion_color

    plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 4, 1); ax.imshow(ct_np, cmap='gray'); ax.set_title("CT"); ax.axis("off")
    ax = plt.subplot(1, 4, 2); ax.imshow(mr_np, cmap='gray'); ax.set_title("MR"); ax.axis("off")

    ax = plt.subplot(1, 4, 3); ax.imshow(ct_np, cmap='gray')
    ax.contour((gt_l  > 0.5).astype(float), levels=[0.5], colors=[liver_color[:3]],  linewidths=1.2)
    ax.contour((gt_le > 0.5).astype(float), levels=[0.5], colors=[lesion_color[:3]], linewidths=1.2)
    ax.set_title(f"CT + GT ({class_labels[0]} red / {class_labels[1]} green)"); ax.axis("off")

    ax = plt.subplot(1, 4, 4); ax.imshow(ct_np, cmap='gray')
    ax.contour((pr_l  > 0.5).astype(float), levels=[0.5], colors=[liver_color[:3]],  linewidths=1.2)
    ax.contour((pr_le > 0.5).astype(float), levels=[0.5], colors=[lesion_color[:3]], linewidths=1.2)
    ax.set_title(f"CT + Pred ({class_labels[0]} red / {class_labels[1]} green)"); ax.axis("off") 

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

def save_epoch_visual(ct, mr, gt, pred_logits, out_png, image_size=1024, class_name="liver"):    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    ct_np = _to_numpy_img(ct)
    mr_np = _to_numpy_img(mr)
    gt_np = gt.detach().cpu().squeeze().numpy()
    pr_np = _to_numpy_mask(pred_logits)

    if class_name == "liver":
        mask_color = [1, 0, 0, 0.6] 
        title_suffix = " (Liver)"
    elif class_name == "lesion":
        mask_color = [0, 1, 0, 0.6]  
        title_suffix = " (Lesion)"
    else:
        mask_color = [0, 0, 1, 0.6]  
        title_suffix = ""

    plt.figure(figsize=(16, 4))
    
    ax = plt.subplot(1, 4, 1)
    ax.imshow(ct_np, cmap='gray')
    ax.set_title("CT")
    ax.axis("off")
    
    ax = plt.subplot(1, 4, 2)
    ax.imshow(mr_np, cmap='gray')
    ax.set_title("MR")
    ax.axis("off")
    
    line_color = mask_color[:3]  # RGBA → RGB

    ax = plt.subplot(1, 4, 3)
    ax.imshow(ct_np, cmap='gray')
    ax.contour((gt_np > 0.5).astype(float), levels=[0.5], colors=[line_color], linewidths=1.2)
    ax.set_title(f"GT{title_suffix}")
    ax.axis("off")

    ax = plt.subplot(1, 4, 4)
    ax.imshow(ct_np, cmap='gray')
    ax.contour((pr_np > 0.5).astype(float), levels=[0.5], colors=[line_color], linewidths=1.2)
    ax.set_title(f"Pred{title_suffix}")
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

def save_gate_curve(history, out_png="outputs/vis/gate_curve.png"):
    if "gate_value" not in history or not history["gate_value"]:
        return
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    epochs = list(range(1, len(history["gate_value"]) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["gate_value"], color="green", linewidth=2, marker="o", markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Fusion gate value", fontsize=12)
    plt.title("Gate value over epochs", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()

# def update_overlay_and_save(overlay_cache, cache_key, ct, gt_vis, pred_vis, class_name, vis_dir, image_size):
#     if cache_key not in overlay_cache:
#         overlay_cache[cache_key] = {"ct": None, "gt_liver": None, "pred_liver": None, "gt_lesion": None, "pred_lesion": None}
#     if overlay_cache[cache_key]["ct"] is None:
#         overlay_cache[cache_key]["ct"] = ct.detach()
#     overlay_cache[cache_key]["gt_" + class_name] = gt_vis.detach()
#     overlay_cache[cache_key]["pred_" + class_name] = pred_vis.detach()

#     entry = overlay_cache[cache_key]
#     if entry["gt_liver"] is not None and entry["gt_lesion"] is not None:
#         pk, si = cache_key
#         case_id_vis = f"{pk}_slice{si:03d}" if si >= 0 else pk
#         out_png = os.path.join(vis_dir, f"case_{case_id_vis}_overlay.png")
#         save_epoch_visual_overlay(
#             entry["ct"], entry["gt_liver"], entry["gt_lesion"],
#             entry["pred_liver"], entry["pred_lesion"],
#             out_png=out_png, image_size=image_size,
#         )
#         del overlay_cache[cache_key]
#         return True
#     return False

def update_overlay_and_save(
    overlay_cache, cache_key, ct, mr, gt_vis, pred_vis, class_name, vis_dir, image_size,
    required_classes=("liver", "lesion"),
):
    if cache_key not in overlay_cache:
        entry = {"ct": None, "mr": None}
        for c in required_classes:                
            entry[f"gt_{c}"] = None
            entry[f"pred_{c}"] = None
        overlay_cache[cache_key] = entry

    if overlay_cache[cache_key]["ct"] is None:
        overlay_cache[cache_key]["ct"] = ct.detach()
    if overlay_cache[cache_key]["mr"] is None:
        overlay_cache[cache_key]["mr"] = mr.detach()
    if class_name in required_classes:                
        overlay_cache[cache_key][f"gt_{class_name}"] = gt_vis.detach()
        overlay_cache[cache_key][f"pred_{class_name}"] = pred_vis.detach()

    entry = overlay_cache[cache_key]
    ready = all(entry.get(f"gt_{c}") is not None for c in required_classes)
    if ready:
        pk, z_native = cache_key
        out_png = os.path.join(vis_dir, f"case_{pk}_z{int(z_native):04d}_overlay.png")
        if len(required_classes) == 1:
            c0 = required_classes[0]
            save_epoch_visual(
                entry["ct"], entry["mr"],
                entry[f"gt_{c0}"], entry[f"pred_{c0}"],
                out_png=out_png, image_size=image_size, class_name=c0,
            )
        else:
            c0, c1 = required_classes[0], required_classes[1]
            save_epoch_visual_overlay(
                entry["ct"], entry["mr"],
                entry[f"gt_{c0}"], entry[f"gt_{c1}"], 
                entry[f"pred_{c0}"], entry[f"pred_{c1}"],
                out_png=out_png, image_size=image_size,
                class_labels=(c0.upper(), c1.upper()),  
            )
        del overlay_cache[cache_key]
        return True
    return False

def save_training_curves(history, out_png="outputs/vis/curves.png"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # dataset 종류 자동 감지
    is_hecktor = bool(history.get("train_dice_tumor"))
    plt.figure(figsize=(18, 5))

    ax = plt.subplot(1, 3, 1)
    ax.plot(epochs, history["train_loss"], label="train", color='blue', linewidth=2)
    ax.plot(epochs, history["val_loss"], label="val", color='red', linewidth=2)
    ax.set_title("Loss", fontsize=12, fontweight='bold')
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = plt.subplot(1, 3, 2)
    ax.plot(epochs, history["train_dice"], label="train dice", color='blue', linestyle='-', linewidth=2)
    ax.plot(epochs, history["val_dice"], label="val dice", color='red', linestyle='-', linewidth=2)
    ax.plot(epochs, history["train_iou"], label="train IoU", color='blue', linestyle='--', linewidth=2)
    ax.plot(epochs, history["val_iou"], label="val IoU", color='red', linestyle='--', linewidth=2)
    ax.set_title("Overall Dice / IoU", fontsize=12, fontweight='bold')
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = plt.subplot(1, 3, 3)
    if is_hecktor:
        n_tr = len(history["train_dice_tumor"])
        n_va = len(history["val_dice_tumor"])
        ep_tr = list(range(len(epochs) - n_tr + 1, len(epochs) + 1))
        ep_va = list(range(len(epochs) - n_va + 1, len(epochs) + 1))
        ax.plot(ep_tr, history["train_dice_tumor"], label="Tumor train dice",
                color='#0066CC', linestyle='-', linewidth=2, marker='o', markersize=4)
        ax.plot(ep_va, history["val_dice_tumor"],   label="Tumor val dice",
                color='#3399FF', linestyle='-', linewidth=2, marker='s', markersize=4)
        ax.plot(ep_tr, history["train_iou_tumor"],  label="Tumor train IoU",
                color='#0066CC', linestyle='--', linewidth=2, marker='o', markersize=4)
        ax.plot(ep_va, history["val_iou_tumor"],    label="Tumor val IoU",
                color='#3399FF', linestyle='--', linewidth=2, marker='s', markersize=4)
        ax.set_title("Tumor Dice & IoU", fontsize=12, fontweight='bold')
    else:
        ax.plot(epochs, history["train_dice_liver"], label="Liver train dice",
                color='#0066CC', linestyle='-', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, history["val_dice_liver"], label="Liver val dice",
                color='#3399FF', linestyle='-', linewidth=2, marker='s', markersize=4)
        ax.plot(epochs, history["train_iou_liver"], label="Liver train IoU",
                color='#0066CC', linestyle='--', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, history["val_iou_liver"], label="Liver val IoU",
                color='#3399FF', linestyle='--', linewidth=2, marker='s', markersize=4)
        ax.plot(epochs, history["train_dice_lesion"], label="Lesion train dice",
                color='#CC0000', linestyle='-', linewidth=2, marker='^', markersize=4)
        ax.plot(epochs, history["val_dice_lesion"], label="Lesion val dice",
                color='#FF6666', linestyle='-', linewidth=2, marker='v', markersize=4)
        ax.plot(epochs, history["train_iou_lesion"], label="Lesion train IoU",
                color='#CC0000', linestyle='--', linewidth=2, marker='^', markersize=4)
        ax.plot(epochs, history["val_iou_lesion"], label="Lesion val IoU",
                color='#FF6666', linestyle='--', linewidth=2, marker='v', markersize=4)
        ax.set_title("Class-wise Dice / IoU", fontsize=12, fontweight='bold')

    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()