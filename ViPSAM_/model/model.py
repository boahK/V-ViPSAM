import torch, torch.nn as nn

from einops import rearrange

from segment_anything import sam_model_registry
from .module import CrossAttentionFusion

class CTMRISegModel(nn.Module):
    def __init__(self, cfg, fusion):
        super().__init__()
        encoder_ct = sam_model_registry["vit_b"](checkpoint=cfg.sam_ckpt, image_size=cfg.image_size, num_frames=cfg.num_frames)
        encoder_mri = sam_model_registry["vit_b"](checkpoint=cfg.sam_ckpt, image_size=cfg.image_size, num_frames=cfg.num_frames)
        
        self.ct_encoder = encoder_ct.image_encoder
        self.prompt_encoder = encoder_ct.prompt_encoder
        self.mask_decoder = encoder_ct.mask_decoder
        self.mri_encoder = encoder_mri.image_encoder

        self.use_mid_fusion = cfg.use_mid_fusion # true/false
        self.fuse_layers = cfg.fuse_layers # list of int [2,5,8]
        # self.mid_fusions = nn.ModuleDict({
        #     str(i) : CrossAttentionFusion(c=cfg.embed_dim, heads=cfg.fusion_heads, num_frames=cfg.num_frames) for i in self.fuse_layers
        # }) # THW crossattention module or max
        self.mid_fusions = nn.ModuleDict({
            str(i) : CrossAttentionFusion(c=cfg.embed_dim, heads=cfg.fusion_heads) for i in self.fuse_layers
        }) # THW crossattention module or max

        self.fusion = fusion
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.register_buffer(
            "pixel_mean", 
            torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", 
            torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        )
        
    def _preprocess_sam(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = x * 255.0
            x = (x - self.pixel_mean) / self.pixel_std
            x = x.view(B, T, C, H, W)
        else:
            x = x * 255.0
            x = (x - self.pixel_mean) / self.pixel_std
        return x

    def _tokens_from_volume(self, enc, x_5d):
        # x_5d: (B, C, T, H, W)
        B, C, T, H, W = x_5d.shape
        x = rearrange(x_5d, "b c t h w -> (b t) c h w")
        x = enc.patch_embed(x)  # (B*T, h, w, D)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed[:, :x.shape[1], :x.shape[2], :]
        if T > 1 and enc.temporal_embedding is not None:
            n1, n2 = x.shape[1], x.shape[2]
            x = rearrange(x, "(b t) n1 n2 d -> (b n1 n2) t d", t=T)
            x = x + enc.temporal_embedding
            x = rearrange(x, "(b n1 n2) t d -> (b t) n1 n2 d", n1=n1, n2=n2)
        return x

    def _encode_with_mid_fusion(self, ct_img_enc, mri_img_enc):
        ct_x = self._tokens_from_volume(self.ct_encoder, ct_img_enc)
        mr_x = self._tokens_from_volume(self.mri_encoder, mri_img_enc)
        for i, (ct_blk, mr_blk) in enumerate(zip(self.ct_encoder.blocks, self.mri_encoder.blocks)):
            ct_x = ct_blk(ct_x)
            mr_x = mr_blk(mr_x)
            if self.use_mid_fusion and i in self.fuse_layers:
                ct_nchw = rearrange(ct_x, "bt h w d -> bt d h w")
                mr_nchw = rearrange(mr_x, "bt h w d -> bt d h w")
                fused_nchw = self.mid_fusions[str(i)](ct_nchw, mr_nchw)
                ct_x = rearrange(fused_nchw, "bt d h w -> bt h w d")
        ct_feat = self.ct_encoder.neck(rearrange(ct_x, "bt h w d -> bt d h w"))
        mr_feat = self.mri_encoder.neck(rearrange(mr_x, "bt h w d -> bt d h w"))
        return ct_feat, mr_feat

    def forward(self, ct_img, mri_img, boxes=None, points=None, point_labels=None, masks=None):
        if ct_img.dim() == 4:
            ct_img = ct_img.unsqueeze(1)   # (B, 1, 3, H, W)
        if mri_img.dim() == 4:
            mri_img = mri_img.unsqueeze(1)

        # B, T, C, H, W = ct_img.shape

        ct_img = self._preprocess_sam(ct_img)   # (B, T, 3, H, W)
        mri_img = self._preprocess_sam(mri_img)

        ct_img_enc = rearrange(ct_img, "b t c h w -> b c t h w")   # (B, 3, T, H, W)
        mri_img_enc = rearrange(mri_img, "b t c h w -> b c t h w")

        # ct_feat = self.ct_encoder(ct_img_enc)        # (B*T, C_out, h, w)
        # mri_feat = self.mri_encoder(mri_img_enc)     # (B*T, C_out, h, w)
        ct_feat, mri_feat = self._encode_with_mid_fusion(ct_img_enc, mri_img_enc)

        # img_embed = self.fusion(ct_feat, mri_feat)   # (B*T, C_out, h, w)

        if boxes is not None:
            boxes = boxes.float() 
            if len(boxes.shape) == 2:
                boxes = boxes.unsqueeze(1) 
        else:
            boxes = None
        
        if points is not None and point_labels is not None:
            point_coords = points 
            point_labels_tensor = point_labels  
        else:
            point_coords = None
            point_labels_tensor = None

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels_tensor) if point_coords is not None else None,
                boxes=boxes,
                masks=masks,
            )

        if ct_img.dim() == 5:
            B, T = ct_img.shape[0], ct_img.shape[1]
            image_pe = self.prompt_encoder.get_dense_pe_with_depth(B * T, T)
        else:
            image_pe = self.prompt_encoder.get_dense_pe()

        logits, iou = self.mask_decoder(
            image_embeddings=ct_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return logits, iou



