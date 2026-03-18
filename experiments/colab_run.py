"""
Colab / Kaggle runner — edit the CONFIG BLOCK below and run this file.

No argparse needed. Just set the variables in the CONFIG BLOCK and execute:
    !python colab_run.py

Or in a notebook cell:
    %run colab_run.py
"""

# ============================================================
#  CONFIG BLOCK — edit these for each experiment
# ============================================================

DATASET    = "firescars"      # "firescars"  |  "burnintensity"
ATTN_TYPE  = "none"           # "none" (baseline)  |  "limix"  |  "mitra"
DATA_PCT   = 5.0             # % of training data: 10 | 20 | 50 | 100
BACKBONE   = "prithvi_eo_v2_300"  # "prithvi_eo_v2_300" | "prithvi_eo_v2_300_tl"
                              # | "prithvi_eo_v2_600" | "prithvi_eo_v2_600_tl"

DATA_ROOT  = "/path/to/dataset"   # <-- SET THIS (HuggingFace dataset folder)
OUTPUT_DIR = "./outputs"

MAX_EPOCHS   = 20
BATCH_SIZE   = 8
LR           = 5e-5
SEED         = 42
NUM_WORKERS  = 2              # keep low on Colab (2 is fine)
ATTN_HEADS   = 2              # heads in channel attention
FREEZE_BACKBONE = False       # True = only train attn + decoder

# ============================================================
#  END CONFIG BLOCK — do not edit below unless you know why
# ============================================================

import os, sys, random, math
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

# Make sure channel_attention.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel_attention import build_channel_attention

# ---- constants ----

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

DATASET_CFGS = {
    "firescars": dict(
        num_classes=2, num_frames=1, loss="ce", ignore_index=-1,
        decoder_channels=256,
        bands=["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"],
        coords_encoding=["time", "location"],
        monitor="val/Multiclass_Jaccard_Index", monitor_mode="max",
        lr=5e-5,
    ),
    "burnintensity": dict(
        num_classes=5, num_frames=3, loss="ce", ignore_index=-1,
        decoder_channels=512,
        bands=["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"],
        coords_encoding=["location"],
        monitor="val/Multiclass_Jaccard_Index", monitor_mode="max",
        lr=1e-5,
        class_weights=[0.018, 0.14, 0.076, 0.11, 0.65],
    ),
}

NUM_SPECTRAL_CHANNELS = 6


# ---- helpers ----

def subset_dataset(dataset, pct, seed):
    from torch.utils.data import Subset
    n = len(dataset)
    k = max(1, math.ceil(n * pct / 100.0))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(n), k))
    return Subset(dataset, idx)


def wrap_patch_embed(patch_embed, channel_attn):
    class _Wrapped(nn.Module):
        def __init__(self, pe, ca):
            super().__init__()
            self.pe = pe
            self.ca = ca
        def forward(self, x, *a, **kw):
            return self.ca(self.pe(x, *a, **kw))
    return _Wrapped(patch_embed, channel_attn)


class MetricsPrinter(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        m = {k: f"{v:.4f}" for k, v in trainer.callback_metrics.items()}
        print(f"\n[epoch {trainer.current_epoch}] " +
              "  |  ".join(f"{k}={v}" for k, v in sorted(m.items())))


# ---- main ----

def run():
    pl.seed_everything(SEED, workers=True)

    cfg = DATASET_CFGS[DATASET]
    embed_dim = BACKBONE_EMBED_DIM[BACKBONE]
    neck_idx  = BACKBONE_INDICES[BACKBONE]
    lr        = LR if LR != 5e-5 else cfg["lr"]

    exp_name = f"{DATASET}__{BACKBONE}__{ATTN_TYPE}__pct{int(DATA_PCT)}"
    out_dir  = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Attention  : {ATTN_TYPE}")
    print(f"  Data %     : {DATA_PCT}%")
    print(f"  Backbone   : {BACKBONE}  (D={embed_dim})")
    print(f"  Dataset    : {DATASET}  T={cfg['num_frames']}, {cfg['num_classes']} classes")
    print(f"{'='*60}\n")

    # 1. Channel attention
    ch_attn = build_channel_attention(ATTN_TYPE, embed_dim, NUM_SPECTRAL_CHANNELS, ATTN_HEADS)
    print(f"Attn params: {sum(p.numel() for p in ch_attn.parameters()):,}")

    # 2. Terratorch model
    from terratorch.models import EncoderDecoderFactory
    from terratorch.tasks import SemanticSegmentationTask
    from terratorch.datasets import HLSBands

    band_enums = [getattr(HLSBands, b) for b in cfg["bands"]]

    model_args = dict(
        backbone=BACKBONE,
        backbone_pretrained=True,
        backbone_bands=band_enums,
        backbone_coords_encoding=cfg["coords_encoding"],
        decoder="UperNetDecoder",
        decoder_channels=cfg["decoder_channels"],
        decoder_scale_modules=True,
        num_classes=cfg["num_classes"],
        rescale=True,
        head_dropout=0.1,
        necks=[
            {"name": "SelectIndices", "indices": neck_idx},
            {"name": "ReshapeTokensToImage",
             **({"effective_time_dim": cfg["num_frames"]} if cfg["num_frames"] > 1 else {})},
        ],
    )
    if cfg["num_frames"] > 1:
        model_args["backbone_num_frames"] = cfg["num_frames"]

    factory = EncoderDecoderFactory()
    model = factory.build_model(task="segmentation", **model_args)

    # 3. Inject channel attention
    if ATTN_TYPE != "none":
        model.encoder.patch_embed = wrap_patch_embed(model.encoder.patch_embed, ch_attn)
        print(f"Injected {ATTN_TYPE} channel attention.")

    if FREEZE_BACKBONE:
        for name, p in model.encoder.named_parameters():
            if "patch_embed.ca" not in name:
                p.requires_grad_(False)
        print("Backbone frozen.")

    task_kwargs = dict(
        model_args=model_args,
        model_factory="EncoderDecoderFactory",
        loss=cfg["loss"],
        ignore_index=cfg["ignore_index"],
        freeze_backbone=FREEZE_BACKBONE,
        freeze_decoder=False,
        model=model,
    )
    if "class_weights" in cfg:
        task_kwargs["class_weights"] = cfg["class_weights"]

    task = SemanticSegmentationTask(**task_kwargs)

    # 4. DataModule
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if DATASET == "firescars":
        from terratorch.datamodules import FireScarsNonGeoDataModule
        dm = FireScarsNonGeoDataModule(
            data_root=DATA_ROOT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            no_data_replace=0,
            no_label_replace=-1,
            use_metadata=True,
            train_transform=A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]),
            val_transform=A.Compose([A.Resize(224, 224), ToTensorV2()]),
            test_transform=A.Compose([A.Resize(224, 224), ToTensorV2()]),
        )
    else:
        from terratorch.datamodules import BurnIntensityNonGeoDataModule
        from terratorch.transforms import (
            FlattenTemporalIntoChannels, UnflattenTemporalFromChannels
        )
        dm = BurnIntensityNonGeoDataModule(
            data_root=DATA_ROOT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            use_metadata=True,
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

    if DATA_PCT < 100.0:
        orig = dm.train_dataset
        dm.train_dataset = subset_dataset(orig, DATA_PCT, SEED)
        print(f"Subset: {len(dm.train_dataset)}/{len(orig)} train samples ({DATA_PCT}%)")

    # 5. Trainer
    logger = TensorBoardLogger(save_dir=out_dir, name="tb")
    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=os.path.join(out_dir, "ckpts"),
            filename="{epoch:02d}_{val_mIoU:.4f}",
            monitor=cfg["monitor"],
            mode=cfg["monitor_mode"],
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(monitor=cfg["monitor"], patience=8, mode=cfg["monitor_mode"]),
        MetricsPrinter(),
    ]

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed" if torch.cuda.is_available() else "32",
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        default_root_dir=out_dir,
    )

    # 6. Optimizer — patch task after construction
    def configure_optimizers(self_task):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self_task.model.parameters()),
            lr=lr, weight_decay=0.05,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    import types
    task.configure_optimizers = types.MethodType(configure_optimizers, task)

    # 7. Fit
    print(f"\nLogs -> {out_dir}\n")
    trainer.fit(task, datamodule=dm)

    # 8. Final metrics
    print("\n[Final Validation Metrics]")
    results = trainer.validate(task, datamodule=dm)

    # Write summary
    summary = os.path.join(out_dir, "results.txt")
    with open(summary, "w") as f:
        f.write(f"experiment  : {exp_name}\n")
        f.write(f"attn_type   : {ATTN_TYPE}\n")
        f.write(f"data_pct    : {DATA_PCT}\n")
        f.write(f"dataset     : {DATASET}\n")
        f.write(f"backbone    : {BACKBONE}\n")
        f.write(f"epochs      : {MAX_EPOCHS}\n")
        f.write(f"attn_params : {sum(p.numel() for p in ch_attn.parameters()):,}\n")
        f.write("\n-- val metrics --\n")
        for d in results:
            for k, v in d.items():
                f.write(f"{k}: {v:.6f}\n")
    print(f"\nSummary -> {summary}")
    return results


if __name__ == "__main__":
    run()
