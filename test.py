import os
import json
from collections import OrderedDict
import torch
import argparse

from configs import load_config, build_cfg_from_checkpoint
from model import CrossAttentionFusion, CTMRISegModel, validation_loop
from data import get_test_loader, get_apis_test_loader, get_hecktor_test_loader
from utils import build_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    j = load_config(args.config, args.override)

    dataset_name = str(j.get("dataset_name", "smc")).lower()

    device = torch.device(f"cuda:{j['gpu_device']}" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(j["checkpoint_path"], map_location=device)
    cfg = build_cfg_from_checkpoint(checkpoint)   
    cfg.sam_ckpt = j.get("sam_ckpt", cfg.sam_ckpt)
    
    fusion = CrossAttentionFusion(c=cfg.out_chans, heads=cfg.fusion_heads)
    model = CTMRISegModel(cfg, fusion).to(device)
    start_epoch = checkpoint.get("epoch", 0)

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()


    if dataset_name == "apis":
        test_loader = get_apis_test_loader(j, cfg)
        test_data_path = os.path.join(j["apis_root"], "test")
    elif dataset_name == "hecktor":                          
        test_loader = get_hecktor_test_loader(j, cfg)        
        test_data_path = j["hecktor_resampled_root"]         
    else:
        test_loader = get_test_loader(
            j["test_ct_mr_root"],
            j["test_mask_root"],
            image_size=cfg.image_size,
            num_frames=cfg.num_frames,
            batch_size=j["batch_size"],
            num_workers=j["num_workers"],
        )
        test_data_path = j["test_ct_mr_root"]
        
    save_dir = os.path.join(j["results_dir"], j.get("model_name"))
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    vis_patient_keys = j.get("vis_patient_keys")
    vis_max_patients = int(j.get("vis_max_patients", 0))
    
    test_results = validation_loop(
        model=model,
        val_loader=test_loader,
        cfg=cfg,
        device=device,
        vis_dir=vis_dir,
        epoch=None,
        save_vis=True,
        dataset_name=dataset_name,
        vis_data_list=test_loader.dataset.data_list,
        vis_patient_keys=vis_patient_keys,
        vis_max_patients=vis_max_patients,
    )

    summary = build_summary(
        j["model_name"], j["checkpoint_path"], start_epoch,
        test_data_path, cfg.image_size, cfg.prompt_type,
        test_results, len(test_loader), dataset_name=dataset_name
    )

    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}\ntest done!\n{'='*60}")
    if dataset_name == "apis":
        print("APIS (Lesion-only) results:")
        print(f"   Lesion: IOU  {test_results.get('iou_lesion', 0):.4f} ± {test_results.get('iou_std_lesion', 0):.4f}")
        print(f"           DICE {test_results.get('dice_lesion', 0):.4f} ± {test_results.get('dice_std_lesion', 0):.4f}")
        print(f"           HD95 {test_results.get('hd95_lesion', 0):.4f} ± {test_results.get('hd95_std_lesion', 0):.4f}")
    elif dataset_name == "hecktor":
        print("HECKTOR (Tumor) results:")
        print(f"   Tumor: IOU  {test_results.get('iou_tumor', 0):.4f} ± {test_results.get('iou_std_tumor', 0):.4f}")
        print(f"          DICE {test_results.get('dice_tumor', 0):.4f} ± {test_results.get('dice_std_tumor', 0):.4f}")
        print(f"          HD95 {test_results.get('hd95_tumor', 0):.4f} ± {test_results.get('hd95_std_tumor', 0):.4f}")
    else:
        print("Each class results:")
        print(f"   Liver:  IOU  {test_results.get('iou_liver', 0):.4f} ± {test_results.get('iou_std_liver', 0):.4f}")
        print(f"           DICE {test_results.get('dice_liver', 0):.4f} ± {test_results.get('dice_std_liver', 0):.4f}")
        print(f"           HD95 {test_results.get('hd95_liver', 0):.4f} ± {test_results.get('hd95_std_liver', 0):.4f}")
        print(f"   Lesion: IOU  {test_results.get('iou_lesion', 0):.4f} ± {test_results.get('iou_std_lesion', 0):.4f}")
        print(f"           DICE {test_results.get('dice_lesion', 0):.4f} ± {test_results.get('dice_std_lesion', 0):.4f}")
        print(f"           HD95 {test_results.get('hd95_lesion', 0):.4f} ± {test_results.get('hd95_std_lesion', 0):.4f}")

if __name__ == "__main__":
    main()

# bash train.sh hecktor > train_hecktor.log 2>&1 &
