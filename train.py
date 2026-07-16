import argparse
import torch
import numpy as np

from collections import Counter

from configs import load_config, build_cfg_from_json, Cfg
from model import CrossAttentionFusion, CTMRISegModel, train_loop
from data import get_train_val_loaders, get_apis_train_val_loaders, get_hecktor_train_val_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="ViPSAM training")
    parser.add_argument("--config", type=str, required=True, help="train config JSON")
    parser.add_argument("--override", type=str, nargs="*", default=[])
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    j = load_config(args.config, args.override)   
    cfg: Cfg = build_cfg_from_json(j)          

    set_seed(cfg.seed)

    device = torch.device(
        f"cuda:{j.get('gpu_device', 0)}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Use device: {device}")

    fusion = CrossAttentionFusion(c=cfg.out_chans, heads=cfg.fusion_heads)
    model = CTMRISegModel(cfg, fusion).to(device)

    # Frozen all parameters
    for p in model.parameters():
        p.requires_grad = False

    # # CA trainable parameters
    # for p in model.fusion.parameters():
    #     p.requires_grad = True

    if hasattr(model, "mid_fusions"):
        for p in model.mid_fusions.parameters():
            p.requires_grad = True

    # Adapter trainable parameters
    adapter_keywords = [
        "Space_Adapter",
        "MLP_Adapter",
        "Space_Queries_Adapter",
        "Cross_Space_Adapter1",
        "Cross_Space_Adapter2",
        "MLP_Queries_Adapter",
        "DMLPAdapter",
        "temporal_embedding",
        "depth_pe",
    ]

    for name, p in model.named_parameters():
        if any(k in name for k in adapter_keywords):
            p.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[Param summary]")
    print(f"  Total params      : {total_params:,}")
    print(f"  Trainable params  : {trainable_params:,} "
          f"({trainable_params / total_params * 100:.2f}% of total)")
    print(f"\n[Trainable parameter names (will be updated by optimizer)]")

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}: {p.numel():,}")
            
    # train_loader, val_loader = get_train_val_loaders(j, cfg)

    dataset_name = str(j.get("dataset_name", "smc")).lower()
    if dataset_name == "apis":
        train_loader, val_loader = get_apis_train_val_loaders(j, cfg)
    elif dataset_name == "hecktor":
        train_loader, val_loader = get_hecktor_train_val_loaders(j, cfg)
    else:
        train_loader, val_loader = get_train_val_loaders(j, cfg)

    train_loop(model, train_loader, val_loader, cfg, j, device)

if __name__ == "__main__":
    main()