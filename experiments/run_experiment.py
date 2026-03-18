"""
Prithvi EO 2.0 + Channel Attention Experiment Runner
=====================================================

Usage (Colab/Kaggle cell):

    !python run_experiment.py \
        --dataset firescars \
        --attn_type none \
        --data_pct 20 \
        --backbone prithvi_eo_v2_300 \
        --data_root /path/to/hls_burn_scars \
        --output_dir /path/to/outputs \
        --max_epochs 20

Arguments:
  --dataset      : firescars | burnintensity
  --attn_type    : none | limix | mitra
  --data_pct     : integer 1–100, percentage of training data to use
  --backbone     : prithvi_eo_v2_300 | prithvi_eo_v2_300_tl | prithvi_eo_v2_600 | prithvi_eo_v2_600_tl
  --data_root    : path to dataset root
  --output_dir   : where to save checkpoints and logs
  --max_epochs   : number of training epochs (default 20)
  --batch_size   : batch size (default 8)
  --lr           : learning rate (default 5e-5)
  --seed         : random seed (default 42)
  --attn_heads   : number of heads for channel attention (default 2)
  --freeze_backbone : flag — if set, freezes Prithvi encoder weights
"""

import argparse
import os
import sys
import random
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

# --- local import (same folder) ---
sys.path.insert(0, os.path.dirname(__file__))
from channel_attention import build_channel_attention


# ============================================================
# Backbone embed_dim lookup (needed to size the attn module)
# ============================================================

BACKBONE_EMBED_DIM = {
    "prithvi_eo_v2_300":    1024,
    "prithvi_eo_v2_300_tl": 1024,
    "prithvi_eo_v2_600":    1280,
    "prithvi_eo_v2_600_tl": 1280,
    "prithvi_vit_100":       768,
}

BACKBONE_INDICES = {
    "prithvi_eo_v2_300":    [5, 11, 17, 23],
    "prithvi_eo_v2_300_tl": [5, 11, 17, 23],
    "prithvi_eo_v2_600":    [7, 15, 23, 31],
    "prithvi_eo_v2_600_tl": [7, 15, 23, 31],
    "prithvi_vit_100":      [2,  5,  8, 11],
}

# ============================================================
# Dataset configs
# ============================================================

DATASET_CONFIGS = {
    "firescars": {
        "datamodule_class": "terratorch.datamodules.FireScarsNonGeoDataModule",
        "num_classes": 2,
        "num_frames": 1,
        "bands": ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"],
        "task": "SemanticSegmentationTask",
        "loss": "ce",
        "ignore_index": -1,
        "metrics": ["val/Multiclass_Jaccard_Index"],
        "monitor": "val/Multiclass_Jaccard_Index",
        "monitor_mode": "max",
        "decoder_channels": 256,
        "use_metadata": True,
        "coords_encoding": ["time", "location"],
    },
    "burnintensity": {
        "datamodule_class": "BurnIntensityNonGeoDataModule",
        "num_classes": 5,
        "num_frames": 3,
        "bands": ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"],
        "task": "SemanticSegmentationTask",
        "loss": "ce",
        "ignore_index": -1,
        "metrics": ["val/Multiclass_Jaccard_Index"],
        "monitor": "val/Multiclass_Jaccard_Index",
        "monitor_mode": "max",
        "decoder_channels": 512,
        "use_metadata": True,
        "coords_encoding": ["location"],
    },
}

NUM_SPECTRAL_CHANNELS = 6  # always 6 HLS bands for both datasets


# ============================================================
# Prithvi + ChannelAttention wrapper
# ============================================================

class PrithviWithChannelAttention(nn.Module):
    """
    Wraps the Prithvi backbone to inject a channel attention module
    right after patch embedding, before the transformer blocks run.

    We hook into the forward by subclassing and overriding patch_embed.
    Since PrithviViT exposes `patch_embed` as a sub-module and then
    runs transformer blocks, we wrap at the backbone level:

        tokens = patch_embed(x)          # (B, N, D)
        tokens = channel_attn(tokens)    # (B, N, D)  <-- injected here
        tokens = transformer_blocks(tokens)
    """

    def __init__(self, backbone: nn.Module, channel_attn: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.channel_attn = channel_attn

        # Patch the patch_embed forward so channel attn runs right after it
        original_patch_embed = backbone.patch_embed

        class PatchEmbedWithAttn(nn.Module):
            def __init__(self, pe, ca):
                super().__init__()
                self.pe = pe
                self.ca = ca

            def forward(self, x, *args, **kwargs):
                tokens = self.pe(x, *args, **kwargs)
                tokens = self.ca(tokens)
                return tokens

        backbone.patch_embed = PatchEmbedWithAttn(original_patch_embed, channel_attn)

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)


# ============================================================
# Subset helper
# ============================================================

def subset_dataset(dataset, pct: float, seed: int):
    """Return a Subset keeping `pct`% of indices, stratified by shuffle."""
    n = len(dataset)
    k = max(1, math.ceil(n * pct / 100.0))
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    return Subset(dataset, sorted(indices))


# ============================================================
# Metrics summary printer (Lightning callback)
# ============================================================

class MetricsPrinter(pl.Callback):
    """Prints a clean metrics table at the end of each validation epoch."""

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {k: f"{v:.4f}" for k, v in trainer.callback_metrics.items()}
        lines = [f"  {k:<45} {v}" for k, v in sorted(metrics.items())]
        print("\n[Metrics @ epoch {}]".format(trainer.current_epoch))
        print("\n".join(lines))


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Prithvi + Channel Attention Experiments")

    # Core experiment knobs — the three things you change per run
    p.add_argument("--attn_type", choices=["none", "limix", "mitra"], default="none",
                   help="Channel attention variant: 'none' (baseline), 'limix', or 'mitra'")
    p.add_argument("--data_pct", type=float, default=20.0,
                   help="Percentage of training data to use (1–100). Default: 20")
    p.add_argument("--dataset", choices=["firescars", "burnintensity"], default="firescars",
                   help="Dataset to train on")

    # Model
    p.add_argument("--backbone", default="prithvi_eo_v2_300",
                   choices=list(BACKBONE_EMBED_DIM),
                   help="Prithvi backbone variant")
    p.add_argument("--attn_heads", type=int, default=2,
                   help="Number of heads for channel attention module (default: 2)")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze Prithvi encoder weights (only train attn + decoder)")

    # Data
    p.add_argument("--data_root", required=True,
                   help="Path to dataset root directory")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2,
                   help="DataLoader workers (keep low on Colab/Kaggle)")

    # Training
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--output_dir", default="./outputs",
                   help="Directory for checkpoints and TensorBoard logs")

    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    pl.seed_everything(args.seed, workers=True)

    cfg = DATASET_CONFIGS[args.dataset]
    embed_dim = BACKBONE_EMBED_DIM[args.backbone]
    neck_indices = BACKBONE_INDICES[args.backbone]

    # Experiment name — easy to read in TensorBoard
    exp_name = f"{args.dataset}__{args.backbone}__{args.attn_type}__pct{int(args.data_pct)}"
    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Attention  : {args.attn_type}")
    print(f"  Data %     : {args.data_pct}%")
    print(f"  Dataset    : {args.dataset}  (T={cfg['num_frames']}, {cfg['num_classes']} classes)")
    print(f"  Backbone   : {args.backbone}  (embed_dim={embed_dim})")
    print(f"  Epochs     : {args.max_epochs}")
    print(f"{'='*60}\n")

    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build channel attention module
    # ------------------------------------------------------------------
    channel_attn = build_channel_attention(
        attn_type=args.attn_type,
        embed_dim=embed_dim,
        num_channels=NUM_SPECTRAL_CHANNELS,
        num_heads=args.attn_heads,
    )
    n_attn_params = sum(p.numel() for p in channel_attn.parameters())
    print(f"Channel attention parameters: {n_attn_params:,}")

    # ------------------------------------------------------------------
    # 2. Build model via terratorch EncoderDecoderFactory
    # ------------------------------------------------------------------
    from terratorch.models import EncoderDecoderFactory
    from terratorch.tasks import SemanticSegmentationTask
    from terratorch.datasets import HLSBands

    band_enums = [getattr(HLSBands, b) for b in cfg["bands"]]

    model_args = dict(
        backbone=args.backbone,
        backbone_pretrained=True,
        backbone_bands=band_enums,
        decoder="UperNetDecoder",
        decoder_channels=cfg["decoder_channels"],
        decoder_scale_modules=True,
        num_classes=cfg["num_classes"],
        rescale=True,
        head_dropout=0.1,
        backbone_coords_encoding=cfg["coords_encoding"],
        necks=[
            {"name": "SelectIndices", "indices": neck_indices},
            {"name": "ReshapeTokensToImage",
             **({"effective_time_dim": cfg["num_frames"]} if cfg["num_frames"] > 1 else {})},
        ],
    )
    if cfg["num_frames"] > 1:
        model_args["backbone_num_frames"] = cfg["num_frames"]

    factory = EncoderDecoderFactory()
    prithvi_model = factory.build_model(
        task="segmentation",
        **model_args,
    )

    # Inject channel attention into the backbone patch embedding
    if args.attn_type != "none":
        backbone = prithvi_model.encoder  # terratorch exposes backbone as .encoder
        backbone.patch_embed = _wrap_patch_embed(backbone.patch_embed, channel_attn)
        print(f"Injected {args.attn_type} channel attention after patch embedding.")

    if args.freeze_backbone:
        for name, param in prithvi_model.encoder.named_parameters():
            # Only freeze transformer blocks; keep channel_attn trainable
            if "patch_embed.ca" not in name:
                param.requires_grad_(False)
        print("Backbone frozen (channel attn + decoder remain trainable).")

    # Wrap in Lightning task
    task = SemanticSegmentationTask(
        model_args=model_args,
        model_factory="EncoderDecoderFactory",
        loss=cfg["loss"],
        ignore_index=cfg["ignore_index"],
        freeze_backbone=args.freeze_backbone,
        freeze_decoder=False,
        model=prithvi_model,
    )

    # ------------------------------------------------------------------
    # 3. DataModule + subset
    # ------------------------------------------------------------------
    import albumentations as A
    from terratorch.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels
    from albumentations.pytorch import ToTensorV2

    if args.dataset == "firescars":
        from terratorch.datamodules import FireScarsNonGeoDataModule
        dm = FireScarsNonGeoDataModule(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            no_data_replace=0,
            no_label_replace=-1,
            use_metadata=cfg["use_metadata"],
            train_transform=A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]),
            val_transform=A.Compose([A.Resize(224, 224), ToTensorV2()]),
            test_transform=A.Compose([A.Resize(224, 224), ToTensorV2()]),
        )
    else:  # burnintensity
        # BurnIntensityNonGeoDataModule must be importable in the runtime env
        from terratorch.datamodules import BurnIntensityNonGeoDataModule
        dm = BurnIntensityNonGeoDataModule(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_metadata=cfg["use_metadata"],
            use_full_data=True,
            train_transform=A.Compose([
                FlattenTemporalIntoChannels(),
                A.Flip(),
                ToTensorV2(),
                UnflattenTemporalFromChannels(n_timesteps=3),
            ]),
            val_transform=A.Compose([
                FlattenTemporalIntoChannels(),
                ToTensorV2(),
                UnflattenTemporalFromChannels(n_timesteps=3),
            ]),
            test_transform=A.Compose([
                FlattenTemporalIntoChannels(),
                ToTensorV2(),
                UnflattenTemporalFromChannels(n_timesteps=3),
            ]),
        )

    dm.setup("fit")

    # Apply data percentage subset to training set
    if args.data_pct < 100.0:
        original_train = dm.train_dataset
        dm.train_dataset = subset_dataset(original_train, args.data_pct, args.seed)
        print(f"Training subset: {len(dm.train_dataset)}/{len(original_train)} samples "
              f"({args.data_pct}%)")

    # ------------------------------------------------------------------
    # 4. Trainer
    # ------------------------------------------------------------------
    logger = TensorBoardLogger(save_dir=output_dir, name="tb_logs")

    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename=f"{exp_name}__{{epoch:02d}}__{{val_mIoU:.4f}}",
            monitor=cfg["monitor"],
            mode=cfg["monitor_mode"],
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg["monitor"],
            patience=8,
            mode=cfg["monitor_mode"],
        ),
        MetricsPrinter(),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if torch.cuda.is_available() else "32"

    trainer = pl.Trainer(
        accelerator=accelerator,
        precision=precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        default_root_dir=output_dir,
        enable_checkpointing=True,
    )

    # ------------------------------------------------------------------
    # 5. Optimizer (passed through Lightning's configure_optimizers)
    #    Override task's default by monkey-patching if needed.
    # ------------------------------------------------------------------
    # terratorch tasks accept optimizer/lr_scheduler in CLI mode.
    # For script mode we set them directly:
    task.hparams_initial = {}
    task._optimizer_cls = torch.optim.AdamW
    task._optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    task._scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
    task._scheduler_kwargs = {"T_max": args.max_epochs}

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print(f"\nStarting training — logs at: {output_dir}\n")
    trainer.fit(task, datamodule=dm)

    # ------------------------------------------------------------------
    # 7. Final validation metrics
    # ------------------------------------------------------------------
    print("\n[Final Validation]")
    val_results = trainer.validate(task, datamodule=dm)
    print(val_results)

    # Save a simple results summary
    summary_path = os.path.join(output_dir, "results.txt")
    with open(summary_path, "w") as f:
        f.write(f"experiment: {exp_name}\n")
        f.write(f"attn_type:  {args.attn_type}\n")
        f.write(f"data_pct:   {args.data_pct}\n")
        f.write(f"dataset:    {args.dataset}\n")
        f.write(f"backbone:   {args.backbone}\n")
        f.write(f"epochs:     {args.max_epochs}\n")
        f.write(f"attn_params:{n_attn_params}\n")
        f.write("\n-- val metrics --\n")
        for d in val_results:
            for k, v in d.items():
                f.write(f"{k}: {v:.6f}\n")
    print(f"Results written to {summary_path}")


# ============================================================
# Patch embed wrapper (used in main)
# ============================================================

def _wrap_patch_embed(patch_embed: nn.Module, channel_attn: nn.Module) -> nn.Module:
    """Wrap an existing patch_embed module to run channel_attn after it."""

    class _WrappedPE(nn.Module):
        def __init__(self, pe, ca):
            super().__init__()
            self.pe = pe
            self.ca = ca

        def forward(self, x, *args, **kwargs):
            tokens = self.pe(x, *args, **kwargs)
            tokens = self.ca(tokens)
            return tokens

    return _WrappedPE(patch_embed, channel_attn)


if __name__ == "__main__":
    main()
