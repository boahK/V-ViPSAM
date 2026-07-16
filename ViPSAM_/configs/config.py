from dataclasses import dataclass, fields

@dataclass
class Cfg:
    image_size: int = 512
    embed_dim: int = 768
    out_chans: int = 256
    fusion_heads: int = 8

    lr: float = 1e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-6       
    scheduler: str = "cosine"  
    warmup_epochs: int = 5 
    epochs: int = 60
    batch_size: int = 1
    val_split: float = 0.1
    use_bce: bool = True
    prompt_type: str = "box"
    seed: int = 42

    # Mid-fusie settings
    use_mid_fusion: bool = True
    fuse_layers: tuple = (2,5,8,11)

    num_frames: int = 4
    overlap: int = 1

    val_freq: int = 5

    sam_ckpt: str = ""


def cfg_from_dict(d: dict) -> Cfg:
    names = {f.name for f in fields(Cfg)}
    kwargs = {k: v for k, v in d.items() if k in names}
    if "medsam_ckpt" in d and "sam_ckpt" not in kwargs:
        kwargs["sam_ckpt"] = d["medsam_ckpt"]
    return Cfg(**kwargs)