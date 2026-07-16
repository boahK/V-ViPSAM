import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from configs import Cfg
from . import validation_loop
from .lr import get_cosine_schedule_with_warmup
from .metrics_loss import seg_loss
from .util import accumulate_case_buffers, compute_3d_metrics_from_buffers
from utils import generate_box_prompt, generate_click_prompt
from utils.visualization import save_training_curves, save_gate_curve

def train_loop(model, train_loader, val_loader, cfg : Cfg, train_cfg : dict, device):
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou":  [], "val_iou":  [],
        "train_hd95": [], "val_hd95": [],
        "train_dice_liver": [], "train_dice_lesion": [],
        "train_iou_liver":  [], "train_iou_lesion":  [],
        "train_hd95_liver": [], "train_hd95_lesion": [],
        "train_dice_tumor":  [], "train_iou_tumor":  [], 
        "train_hd95_tumor":  [], "val_dice_tumor":    [],
        "val_iou_tumor":    [], "val_hd95_tumor":    [],
        "val_dice_liver":   [], "val_dice_lesion":   [],
        "val_iou_liver":    [], "val_iou_lesion":    [],
        "val_hd95_liver":   [], "val_hd95_lesion":   [], 
        "gate_value": [],
    }

    print(f"\n{'='*60}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Batch size:    {cfg.batch_size}")
    print(f"  Prompt type:   {cfg.prompt_type}")
    print(f"{'='*60}\n")

    # trainable = list(model.fusion.parameters())
    # if cfg.use_lora:
    #     lora_params = get_lora_params(model.mask_decoder)
    #     params = [
    #         {"params": trainable, "lr": cfg.lr},
    #         {"params": lora_params, "lr": cfg.lr * 0.1},
    #     ]
    # else:
    #     params = trainable
    # opt = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_in_opt = sum(p.numel() for group in opt.param_groups for p in group["params"])
    print(f"[Optimizer] parameters in optimizer: {n_in_opt:,}")


    start_epoch = 0
    best_dice = 0.0
    best_train_dice = 0.0 

    resume_path = train_cfg.get("resume_path", "")
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        history["train_loss"] = checkpoint.get("history_train_loss", [])
        history["val_loss"]   = checkpoint.get("history_val_loss", [])
        history["train_dice"] = checkpoint.get("history_train_dice", [])
        history["val_dice"]   = checkpoint.get("history_val_dice", [])
        history["train_iou"]  = checkpoint.get("history_train_iou", [])
        history["val_iou"]    = checkpoint.get("history_val_iou", [])
        history["train_hd95"] = checkpoint.get("history_train_hd95", [])
        history["val_hd95"] = checkpoint.get("history_val_hd95", [])
        history["train_dice_liver"] = checkpoint.get("history_train_dice_liver", [])
        history["train_dice_lesion"] = checkpoint.get("history_train_dice_lesion", [])
        history["train_iou_liver"] = checkpoint.get("history_train_iou_liver", [])
        history["train_iou_lesion"] = checkpoint.get("history_train_iou_lesion", [])
        history["train_hd95_liver"] = checkpoint.get("history_train_hd95_liver", [])
        history["train_hd95_lesion"] = checkpoint.get("history_train_hd95_lesion", [])
        history["val_dice_liver"] = checkpoint.get("history_val_dice_liver", [])
        history["val_dice_lesion"] = checkpoint.get("history_val_dice_lesion", [])
        history["val_iou_liver"] = checkpoint.get("history_val_iou_liver", [])
        history["val_iou_lesion"] = checkpoint.get("history_val_iou_lesion", [])
        history["val_hd95_liver"] = checkpoint.get("history_val_hd95_liver", [])
        history["val_hd95_lesion"] = checkpoint.get("history_val_hd95_lesion", [])
        history["gate_value"] = checkpoint.get("history_gate_value", [])
        best_dice   = checkpoint.get("best_dice", 0.0)
        best_train_dice = checkpoint.get("best_train_dice", 0.0)  
        start_epoch = checkpoint.get("epoch", 0)
        print(f"[Resume] epoch {start_epoch}")

    num_training_steps = len(train_loader) * cfg.epochs
    warmup_epochs = cfg.warmup_epochs
    num_warmup_steps = warmup_epochs * len(train_loader)
    scheduler = None
    if cfg.scheduler == "cosine":
        last_epoch = start_epoch * len(train_loader) - 1 if start_epoch > 0 else -1
        eta_min_ratio = cfg.min_lr / cfg.lr
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            eta_min_ratio=eta_min_ratio,
            last_epoch=last_epoch,
        )
        if resume_path and os.path.isfile(resume_path) and "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"base lr: {cfg.lr}")
    print(f"scheduler: {cfg.scheduler} (get_cosine_schedule_with_warmup, warmup_epochs={warmup_epochs})")
    print(f"min_lr: {cfg.min_lr}")

    base_dir = os.path.join(train_cfg["output_dir"], train_cfg["experiment_name"])
    ckpt_dir = os.path.join(base_dir, "ckpt")
    vis_dir = os.path.join(base_dir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    image_size = getattr(model,'image_size', cfg.image_size)
    val_freq = cfg.val_freq
    idx_c = 0
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tr_loss = 0.0
        case_buffers = {}
        gaussian_1d = None

        for batch_idx, batch in enumerate(train_loader):
            ct = batch["ct"].to(device)
            mr = batch["mr"].to(device)
            gt = batch["mask"].float().to(device)
            spacing = batch["spacing"].to(device)
            class_names = batch.get("class_name", [])

            B, T, _, H, W = ct.shape
            if epoch == 0 and idx_c < 10:
                idx_c += 1
                print("[Train] epoch 0: ct", ct.shape, "mr", mr.shape, "gt", gt.shape, "B,T=", B, T)
            # if batch_idx == 0:
            #     print(f"[DEBUG][epoch {epoch}] case_id[0]={batch['case_id'][0]}")
            #     print(f"[DEBUG][epoch {epoch}] z_indices[0]={batch['z_indices'][0].tolist()}")

            # GT (B, T, H, W) -> (B*T, 1, H, W)
            gt_bt = gt.unsqueeze(2).view(B * T, 1, H, W)

            spacing_bt = spacing.unsqueeze(1).expand(B, T, 2).reshape(B * T, 2)

            if cfg.prompt_type == "box":
                boxes = generate_box_prompt(gt_bt, bbox_shift=20, image_size=image_size).to(device)
                points = None
                point_labels = None
                # gt: (B, T, H, W), gt_bt: (B*T, 1, H, W)
                # center_t = T // 2                     # 짝수 T면 뒤쪽 중앙 인덱스
                # gt_center = gt[:, center_t:center_t+1, :, :]   # (B,1,H,W)
                # boxes_center = generate_box_prompt(
                #     gt_center,
                #     bbox_shift=20,                 # trainer/tester에서 기존 값 유지
                #     image_size=image_size,
                # ).to(device)                          # (B,4)
                # # 각 샘플의 box를 T프레임에 동일 복제 -> (B*T,4)
                # boxes = boxes_center.repeat_interleave(T, dim=0)
                # points, point_labels = None, None
            else:  
                points, point_labels = generate_click_prompt(gt_bt, pt_label=1, image_size=image_size)
                points = points.to(device)
                point_labels = point_labels.to(device)
                boxes = None
            
            # if epoch == start_epoch and batch_idx == 0:
            #     print(f"\n[Training] frist batch:")
            #     print(f"  Batch size: {ct.shape[0]}")
            #     print(f"  Class names: {class_names}")

            logits, _ = model(ct, mr, boxes=boxes, points=points, point_labels=point_labels, masks=None)
            logits_up = F.interpolate(logits, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)
            # logits_up = F.interpolate(logits, size=(cfg.image_size, cfg.image_size), mode="nearest")
            
            loss = seg_loss(logits_up, gt_bt, use_bce=cfg.use_bce)

            if epoch == 0 and idx_c < 10:
                print("[Train] epoch 0: Train_logits_size", logits_up.shape, "Train_gt_size", gt_bt.shape)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            tr_loss += loss.item()
            gaussian_1d = accumulate_case_buffers(batch, logits_up.detach(), case_buffers, gaussian_1d, device)

        tr_loss /= max(1, len(train_loader))
        train_metrics = compute_3d_metrics_from_buffers(case_buffers)
        tr_dice = train_metrics["dice"]
        tr_iou = train_metrics["iou"]
        tr_hd95 = train_metrics["hd95"]
        tr_dice_liver = train_metrics["dice_liver"]
        tr_iou_liver = train_metrics["iou_liver"]
        tr_hd95_liver = train_metrics["hd95_liver"]
        tr_dice_lesion = train_metrics["dice_lesion"]
        tr_iou_lesion = train_metrics["iou_lesion"]
        tr_hd95_lesion = train_metrics["hd95_lesion"]
        gate_val = model.mid_fusions[str(cfg.fuse_layers[-1])].gate.item()
        history["gate_value"].append(gate_val)

        print("=" * 60)
        print(f"Epoch {epoch+1}/{cfg.epochs} train")
        print("=" * 60)
        print(f">Fusion T-Gate value: {gate_val:.4f}")  
        if epoch > start_epoch:
            gate_change = gate_val - history["gate_value"][-2]
            print(f"   T-Gate_change: {gate_change:+.4f}")
        print(f"total: Dice={tr_dice:.3f}, IoU={tr_iou:.3f}, HD95={tr_hd95:.3f}")
        dataset_name_tr = str(train_cfg.get("dataset_name", "smc")).lower()
        if dataset_name_tr == "hecktor":
            tr_dice_tumor = train_metrics.get("dice_tumor", 0.0)
            tr_iou_tumor  = train_metrics.get("iou_tumor",  0.0)
            tr_hd95_tumor = train_metrics.get("hd95_tumor", 0.0)
            print(f"Tumor: Dice={tr_dice_tumor:.3f}, IoU={tr_iou_tumor:.3f}, HD95={tr_hd95_tumor:.3f}")
        else:
            print(f"Liver:  Dice={tr_dice_liver:.3f},  IoU={tr_iou_liver:.3f},  HD95={tr_hd95_liver:.3f}")
            print(f"Lesion: Dice={tr_dice_lesion:.3f}, IoU={tr_iou_lesion:.3f}, HD95={tr_hd95_lesion:.3f}")

        if tr_dice > best_train_dice:
            best_train_dice = tr_dice
            _train_best_ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch + 1,
                "best_train_dice": best_train_dice,
                "train_dice": tr_dice,
                "train_loss": tr_loss,
            }
            if scheduler is not None:
                _train_best_ckpt["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(_train_best_ckpt, f"{ckpt_dir}/train_best.pth")
            print(f"[Train Best] saved! train_dice={best_train_dice:.4f}")

        do_val = ((epoch + 1) % val_freq == 0) or ((epoch + 1) == cfg.epochs)
        if do_val:
            print("=" * 60)
            print("Validation starts..")
            print("=" * 60)
            val_results = validation_loop(
                model=model,
                val_loader=val_loader,
                cfg=cfg,
                device=device,
                vis_dir=vis_dir,
                epoch=epoch+1,
                save_vis=True,
                dataset_name=train_cfg.get("dataset_name", "smc"),
            )
            va_loss   = val_results["loss"]
            va_dice   = val_results["dice"]
            va_iou    = val_results["iou"]
            va_hd95   = val_results["hd95"]
            va_dice_liver  = val_results.get("dice_liver",  0.0)
            va_iou_liver   = val_results.get("iou_liver",   0.0)
            va_hd95_liver  = val_results.get("hd95_liver",  0.0)
            va_dice_lesion = val_results.get("dice_lesion", 0.0)
            va_iou_lesion  = val_results.get("iou_lesion",  0.0)
            va_hd95_lesion = val_results.get("hd95_lesion", 0.0)
            if dataset_name_tr == "hecktor":
                va_dice_tumor = val_results.get("dice_tumor", 0.0)
                va_iou_tumor  = val_results.get("iou_tumor",  0.0)
                va_hd95_tumor = val_results.get("hd95_tumor", 0.0)
            else:
                va_dice_tumor = va_iou_tumor = va_hd95_tumor = 0.0
        else:
            import math
            _nan = float('nan')
            va_loss = va_dice = va_iou = va_hd95 = _nan
            va_dice_liver = va_iou_liver = va_hd95_liver = _nan
            va_dice_lesion = va_iou_lesion = va_hd95_lesion = _nan
            va_dice_tumor = va_iou_tumor = va_hd95_tumor = _nan


        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(va_iou)
        history["train_hd95"].append(tr_hd95) 
        history["val_hd95"].append(va_hd95)    
        history["train_dice_liver"].append(tr_dice_liver)
        history["train_dice_lesion"].append(tr_dice_lesion)
        history["train_iou_liver"].append(tr_iou_liver)
        history["train_iou_lesion"].append(tr_iou_lesion)
        history["train_hd95_liver"].append(tr_hd95_liver)  
        history["train_hd95_lesion"].append(tr_hd95_lesion) 
        history["val_dice_liver"].append(va_dice_liver)
        history["val_dice_lesion"].append(va_dice_lesion)
        history["val_iou_liver"].append(va_iou_liver)
        history["val_iou_lesion"].append(va_iou_lesion)
        history["val_hd95_liver"].append(va_hd95_liver) 
        history["val_hd95_lesion"].append(va_hd95_lesion) 
        if dataset_name_tr == "hecktor":
            history["train_dice_tumor"].append(tr_dice_tumor)
            history["train_iou_tumor"].append(tr_iou_tumor)
            history["train_hd95_tumor"].append(tr_hd95_tumor)
            history["val_dice_tumor"].append(va_dice_tumor)
            history["val_iou_tumor"].append(va_iou_tumor)
            history["val_hd95_tumor"].append(va_hd95_tumor)

        current_lr = opt.param_groups[0]["lr"]
        print(f"  lr: {current_lr:.6e}")

        save_training_curves(history, out_png=f"{vis_dir}/curves.png")
        save_gate_curve(history, out_png=f"{vis_dir}/gate_curve.png")

        print(f"[{epoch+1}/{cfg.epochs}] loss={tr_loss:.4f}")
        print(f"  Total: tr(D/I/HD95)={tr_dice:.3f}/{tr_iou:.3f}/{tr_hd95:.2f} | "
            f"val(D/I/HD95)={va_dice:.3f}/{va_iou:.3f}/{va_hd95:.2f}")
        if dataset_name_tr == "hecktor":
            print(f"  Tumor: tr(D/I/HD95)={tr_dice_tumor:.3f}/{tr_iou_tumor:.3f}/{tr_hd95_tumor:.2f} | "
                f"val(D/I/HD95)={va_dice_tumor:.3f}/{va_iou_tumor:.3f}/{va_hd95_tumor:.2f}")
        else:
            print(f"  Liver:  tr(D/I/HD95)={tr_dice_liver:.3f}/{tr_iou_liver:.3f}/{tr_hd95_liver:.2f} | "
                f"val(D/I/HD95)={va_dice_liver:.3f}/{va_iou_liver:.3f}/{va_hd95_liver:.2f}")
            print(f"  Lesion: tr(D/I/HD95)={tr_dice_lesion:.3f}/{tr_iou_lesion:.3f}/{tr_hd95_lesion:.2f} | "
                f"val(D/I/HD95)={va_dice_lesion:.3f}/{va_iou_lesion:.3f}/{va_hd95_lesion:.2f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch + 1,
            "best_dice": best_dice,
            "best_train_dice": best_train_dice,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_dice": tr_dice,
            "val_dice": va_dice,
            "train_hd95": tr_hd95,
            "val_hd95": va_hd95,
            "history_train_loss": history["train_loss"],
            "history_val_loss": history["val_loss"],
            "history_train_dice": history["train_dice"],
            "history_val_dice": history["val_dice"],
            "history_train_iou": history["train_iou"],
            "history_val_iou": history["val_iou"],
            "history_train_hd95": history["train_hd95"],
            "history_val_hd95": history["val_hd95"],
            "history_train_dice_liver": history["train_dice_liver"],
            "history_train_dice_lesion": history["train_dice_lesion"],
            "history_train_iou_liver": history["train_iou_liver"],
            "history_train_iou_lesion": history["train_iou_lesion"],
            "history_train_hd95_liver": history["train_hd95_liver"],
            "history_train_hd95_lesion": history["train_hd95_lesion"],
            "history_val_dice_liver": history["val_dice_liver"],
            "history_val_dice_lesion": history["val_dice_lesion"],
            "history_val_iou_liver": history["val_iou_liver"],
            "history_val_iou_lesion": history["val_iou_lesion"],
            "history_val_hd95_liver": history["val_hd95_liver"],
            "history_val_hd95_lesion": history["val_hd95_lesion"],
            "history_train_dice_tumor": history["train_dice_tumor"],
            "history_train_iou_tumor":  history["train_iou_tumor"],
            "history_train_hd95_tumor": history["train_hd95_tumor"],
            "history_val_dice_tumor":   history["val_dice_tumor"],
            "history_val_iou_tumor":    history["val_iou_tumor"],
            "history_val_hd95_tumor":   history["val_hd95_tumor"],
            "config": cfg.__dict__,
            "prompt_type": cfg.prompt_type,
            "history_gate_value": history["gate_value"],
        }
        
        if do_val and va_dice > best_dice:
            best_dice = va_dice
            checkpoint["best_dice"] = best_dice
            torch.save(checkpoint, f"{ckpt_dir}/best.pth")
            print(f"~~~~~~~~~~~~Best model saved! Dice: {best_dice:.4f}")

        if (epoch + 1) % 3 == 0:
            torch.save(checkpoint, f"{ckpt_dir}/epoch_{epoch+1:03d}.pth")

        torch.save(checkpoint, f"{ckpt_dir}/latest.pth")

