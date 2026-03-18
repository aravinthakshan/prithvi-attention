"""
Colab / Kaggle runner — edit the CONFIG BLOCK below and run this file.

    !python colab_run.py
    OR
    %run colab_run.py
"""

# ============================================================
#  CONFIG BLOCK — edit these for each experiment
# ============================================================

DATASET    = "firescars"           # "firescars"  |  "burnintensity"
ATTN_TYPE  = "none"                # "none" (baseline)  |  "limix"  |  "mitra"
DATA_PCT   = 100.0                 # % of training data: 5 | 10 | 20 | 50 | 100
BACKBONE   = "prithvi_eo_v2_300_tl"

DATA_ROOT  = None                  # None = auto-download  |  "/your/path" = skip download
OUTPUT_DIR = "./outputs"
HF_TOKEN   = "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"

MAX_EPOCHS   = 20
BATCH_SIZE   = 8
LR           = 5e-5
SEED         = 42
ATTN_HEADS   = 2
FREEZE_BACKBONE = False

# ============================================================
#  END CONFIG BLOCK
# ============================================================

import os, sys, random, math, time, logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
import channel_attention as _ca_mod
importlib.reload(_ca_mod)
from channel_attention import build_channel_attention

logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("terratorch").setLevel(logging.WARNING)

def ts():
    return datetime.now().strftime('%H:%M:%S')

# ---- Dataset download/extract helpers ----

HF_DATASET_IDS = {
    "firescars":     "ibm-nasa-geospatial/hls_burn_scars",
    "burnintensity": "ibm-nasa-geospatial/burn_intensity",
}

def get_data_root(dataset, data_root, output_dir):
    from huggingface_hub import snapshot_download
    if data_root and os.path.isdir(data_root):
        _extract(data_root)
        return _resolve(data_root)
    dest = os.path.join(output_dir, "data", dataset)
    if not (os.path.isdir(dest) and os.listdir(dest)):
        print(f"[{ts()}] Downloading {HF_DATASET_IDS[dataset]}...", flush=True)
        os.makedirs(dest, exist_ok=True)
        snapshot_download(repo_id=HF_DATASET_IDS[dataset], repo_type="dataset",
                          local_dir=dest, token=HF_TOKEN,
                          ignore_patterns=["*.md", "*.gitattributes"])
    else:
        print(f"[{ts()}] Dataset at {dest}", flush=True)
    _extract(dest)
    return _resolve(dest)

def _extract(base):
    import tarfile
    if any(os.path.isdir(os.path.join(base, d)) for d in ("training","validation","train","val")):
        return
    for f in sorted(os.listdir(base)):
        if f.endswith((".tar.gz",".tgz",".tar")):
            print(f"[{ts()}] Extracting {f}...", flush=True)
            with tarfile.open(os.path.join(base, f), "r:*") as t:
                t.extractall(path=base, filter="data")
            return

def _resolve(base):
    markers = {"training","validation","train","val","test"}
    for root, dirs, files in os.walk(base):
        if root[len(base):].count(os.sep) > 3: break
        if {d.lower() for d in dirs} & markers: return root
        if any(f.endswith(".tif") for f in files): return root
    return base

# ---- Config ----

CFGS = {
    "firescars": dict(
        num_classes=2, num_frames=1, loss="ce", ignore_index=-1,
        decoder_channels=256,
        bands=["BLUE","GREEN","RED","NIR_BROAD","SWIR_1","SWIR_2"],
        coords=["time","location"],
        monitor="val/Multiclass_Jaccard_Index", mode="max",
    ),
    "burnintensity": dict(
        num_classes=5, num_frames=3, loss="ce", ignore_index=-1,
        decoder_channels=512,
        bands=["BLUE","GREEN","RED","NIR","SWIR_1","SWIR_2"],
        coords=["location"],
        monitor="val/Multiclass_Jaccard_Index", mode="max",
        class_weights=[0.018, 0.14, 0.076, 0.11, 0.65],
    ),
}

NECK_IDX = {
    "prithvi_eo_v2_300": [5,11,17,23], "prithvi_eo_v2_300_tl": [5,11,17,23],
    "prithvi_eo_v2_600": [7,15,23,31], "prithvi_eo_v2_600_tl": [7,15,23,31],
}
EMBED_DIM = {
    "prithvi_eo_v2_300": 1024, "prithvi_eo_v2_300_tl": 1024,
    "prithvi_eo_v2_600": 1280, "prithvi_eo_v2_600_tl": 1280,
}

# ---- Minimal Lightning module (bypasses terratorch task issues) ----

class PrithviSegTask(torch.nn.Module):
    """Thin wrapper: Prithvi encoder-decoder + loss + metrics, no terratorch task."""
    pass

import lightning.pytorch as pl
from torchmetrics.classification import MulticlassJaccardIndex

class LitPrithviSeg(pl.LightningModule):
    def __init__(self, model, ch_attn, num_classes, ignore_index, lr, class_weights=None):
        super().__init__()
        self.model = model
        self.ch_attn = ch_attn
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, phase):
        img = batch["image"]  # (B, C, T, H, W) or (B, C, H, W)
        mask = batch["mask"]  # (B, H, W) or (B, 1, H, W)

        if img.ndim == 4:
            img = img.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        if mask.ndim == 4:
            mask = mask.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

        out = self.model(img)  # terratorch model output
        # terratorch models return a ModelOutput with .output
        if hasattr(out, 'output'):
            logits = out.output
        elif isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out

        # logits: (B, num_classes, H, W)
        loss = F.cross_entropy(logits, mask.long(),
                               weight=self.class_weights,
                               ignore_index=self.ignore_index)

        if phase == "val":
            preds = logits.argmax(dim=1)
            self.val_iou.update(preds, mask.long())

        self.log(f"{phase}/loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_validation_epoch_end(self):
        iou = self.val_iou.compute()
        self.log("val/Multiclass_Jaccard_Index", iou, prog_bar=False)
        self.val_iou.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr, weight_decay=0.05
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ---- DataModule helper ----

def make_datamodule(dataset, data_root, batch_size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    nw = 2  # Colab-safe; set to 4 on Kaggle/local

    if dataset == "firescars":
        from terratorch.datamodules import FireScarsNonGeoDataModule
        return FireScarsNonGeoDataModule(
            data_root=data_root, batch_size=batch_size, num_workers=nw,
            no_data_replace=0, no_label_replace=-1, use_metadata=True,
            train_transform=A.Compose([A.Resize(224,224), A.HorizontalFlip(p=0.5),
                                       A.VerticalFlip(p=0.5), ToTensorV2()]),
            val_transform=A.Compose([A.Resize(224,224), ToTensorV2()]),
            test_transform=A.Compose([A.Resize(224,224), ToTensorV2()]),
        )
    else:
        from terratorch.datamodules import BurnIntensityNonGeoDataModule
        from terratorch.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels
        return BurnIntensityNonGeoDataModule(
            data_root=data_root, batch_size=batch_size, num_workers=nw,
            use_metadata=True, use_full_data=True,
            train_transform=A.Compose([FlattenTemporalIntoChannels(), A.Flip(), ToTensorV2(),
                                       UnflattenTemporalFromChannels(n_timesteps=3)]),
            val_transform=A.Compose([FlattenTemporalIntoChannels(), ToTensorV2(),
                                     UnflattenTemporalFromChannels(n_timesteps=3)]),
            test_transform=A.Compose([FlattenTemporalIntoChannels(), ToTensorV2(),
                                      UnflattenTemporalFromChannels(n_timesteps=3)]),
        )


# ---- Subset ----

def subset_dataset(ds, pct, seed):
    n = len(ds)
    k = max(1, min(n, math.ceil(n * pct / 100.0)))
    idx = sorted(random.Random(seed).sample(range(n), k))
    return Subset(ds, idx)


# ---- Patch embed wrapper ----

def wrap_patch_embed(pe, ca):
    """Wrap patch_embed to run channel_attn after it, while proxying all
    attributes (grid_size, num_patches, etc.) from the original patch_embed."""
    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.pe = pe
            self.ca = ca
        def forward(self, x, *a, **kw):
            return self.ca(self.pe(x, *a, **kw))
        def __getattr__(self, name):
            # First check our own attrs (pe, ca, etc.)
            try:
                return super().__getattr__(name)
            except AttributeError:
                # Fall through to the original patch_embed for grid_size, num_patches, etc.
                return getattr(self.pe, name)
    return W()


# ============================================================
# Main
# ============================================================

def run():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    pl.seed_everything(SEED, workers=True)

    cfg = CFGS[DATASET]
    embed_dim = EMBED_DIM[BACKBONE]
    neck_idx = NECK_IDX[BACKBONE]

    exp_name = f"{DATASET}__{BACKBONE}__{ATTN_TYPE}__pct{int(DATA_PCT)}"
    out_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Data
    data_root = get_data_root(DATASET, DATA_ROOT, OUTPUT_DIR)
    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Attention  : {ATTN_TYPE}")
    print(f"  Data %     : {DATA_PCT}%")
    print(f"  Backbone   : {BACKBONE}  (D={embed_dim})")
    print(f"  Dataset    : {DATASET}  T={cfg['num_frames']}, {cfg['num_classes']} classes")
    print(f"{'='*60}\n")

    # 2. Build model via terratorch factory
    print(f"[{ts()}] Building model...", flush=True)
    from terratorch.models import EncoderDecoderFactory
    from terratorch.datasets import HLSBands

    band_enums = [getattr(HLSBands, b) for b in cfg["bands"]]
    is_tl = BACKBONE.endswith("_tl")

    model_args = dict(
        backbone=BACKBONE, backbone_pretrained=True,
        backbone_bands=band_enums,
        backbone_coords_encoding=cfg["coords"] if is_tl else [],
        decoder="UperNetDecoder", decoder_channels=cfg["decoder_channels"],
        decoder_scale_modules=True, num_classes=cfg["num_classes"],
        rescale=True, head_dropout=0.1,
        necks=[{"name": "SelectIndices", "indices": neck_idx},
               {"name": "ReshapeTokensToImage",
                **({"effective_time_dim": cfg["num_frames"]} if cfg["num_frames"] > 1 else {})}],
    )
    if cfg["num_frames"] > 1:
        model_args["backbone_num_frames"] = cfg["num_frames"]

    model = EncoderDecoderFactory().build_model(task="segmentation", **model_args)
    print(f"[{ts()}] Model built. Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # 3. Channel attention
    ch_attn = build_channel_attention(ATTN_TYPE, embed_dim, 6, ATTN_HEADS)
    attn_params = sum(p.numel() for p in ch_attn.parameters())
    print(f"[{ts()}] Channel attn: {ATTN_TYPE} ({attn_params:,} params)", flush=True)

    if ATTN_TYPE != "none":
        model.encoder.patch_embed = wrap_patch_embed(model.encoder.patch_embed, ch_attn)
        print(f"[{ts()}] Injected into patch_embed.", flush=True)

    if FREEZE_BACKBONE:
        for n, p in model.encoder.named_parameters():
            if "patch_embed.ca" not in n:
                p.requires_grad_(False)
        print(f"[{ts()}] Backbone frozen.", flush=True)

    # 4. Wrap in clean Lightning module (no terratorch task overhead)
    print(f"[{ts()}] Building Lightning module...", flush=True)
    lit = LitPrithviSeg(
        model=model, ch_attn=ch_attn,
        num_classes=cfg["num_classes"],
        ignore_index=cfg["ignore_index"],
        lr=LR,
        class_weights=cfg.get("class_weights"),
    )
    print(f"[{ts()}] Lightning module ready.", flush=True)

    # 5. DataModule
    print(f"[{ts()}] Setting up data...", flush=True)
    dm = make_datamodule(DATASET, data_root, BATCH_SIZE)
    dm.setup("fit")
    print(f"[{ts()}] Train samples: {len(dm.train_dataset)}", flush=True)

    if DATA_PCT < 100.0:
        orig = dm.train_dataset
        dm.train_dataset = subset_dataset(orig, DATA_PCT, SEED)
        print(f"[{ts()}] Subset: {len(dm.train_dataset)}/{len(orig)} ({DATA_PCT}%)", flush=True)

    # 6. Quick sanity: run one batch manually
    print(f"[{ts()}] Sanity check: loading 1 batch...", flush=True)
    dl = DataLoader(dm.train_dataset, batch_size=min(4, len(dm.train_dataset)),
                    shuffle=False, num_workers=0)
    batch = next(iter(dl))
    print(f"[{ts()}] Batch loaded — image: {batch['image'].shape}, mask: {batch['mask'].shape}", flush=True)

    print(f"[{ts()}] Sanity check: forward pass...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = batch["image"].to(device)
        # Prithvi expects (B, C, T, H, W) — add T dim if missing
        if img.ndim == 4:
            img = img.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        print(f"[{ts()}] Forward input shape: {img.shape}", flush=True)
        out = model(img)
    if hasattr(out, 'output'):
        print(f"[{ts()}] Forward OK — output: {out.output.shape}", flush=True)
    elif isinstance(out, (list, tuple)):
        print(f"[{ts()}] Forward OK — output: {out[0].shape}", flush=True)
    else:
        print(f"[{ts()}] Forward OK — output: {out.shape}", flush=True)
    model.cpu()  # move back so Lightning handles placement
    model.train()

    # 7. Trainer
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    class EpochPrinter(pl.Callback):
        def __init__(self): super().__init__(); self._t = None
        def on_train_epoch_start(self, trainer, pl_module):
            self._t = time.time()
            print(f"[{ts()}] epoch {trainer.current_epoch+1}/{trainer.max_epochs} train...", flush=True)
        def on_train_epoch_end(self, trainer, pl_module):
            e = time.time() - self._t if self._t else 0
            print(f"[{ts()}] epoch {trainer.current_epoch+1}/{trainer.max_epochs} train done ({e:.1f}s)", flush=True)
        def on_validation_epoch_start(self, trainer, pl_module):
            print(f"[{ts()}] validating...", flush=True)
        def on_validation_epoch_end(self, trainer, pl_module):
            m = {k: f"{v:.4f}" for k, v in trainer.callback_metrics.items() if "val" in k}
            print(f"[{ts()}] {' | '.join(f'{k}={v}' for k,v in sorted(m.items()))}", flush=True)
        def on_train_end(self, trainer, pl_module):
            print(f"[{ts()}] Training complete.", flush=True)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed" if torch.cuda.is_available() else "32",
        max_epochs=MAX_EPOCHS,
        callbacks=[
            EpochPrinter(),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(dirpath=os.path.join(out_dir, "ckpts"), filename="epoch{epoch:02d}",
                            monitor=cfg["monitor"], mode=cfg["mode"], save_top_k=1, save_last=True),
            EarlyStopping(monitor=cfg["monitor"], patience=8, mode=cfg["mode"]),
        ],
        logger=TensorBoardLogger(save_dir=out_dir, name="tb"),
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        default_root_dir=out_dir,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    # 8. Train
    print(f"\n[{ts()}] Starting training...\n", flush=True)
    t_start = time.time()
    trainer.fit(lit, datamodule=dm)
    t_total = time.time() - t_start
    print(f"[{ts()}] Training finished in {t_total:.0f}s ({t_total/60:.1f} min)", flush=True)

    # 9. Final metrics
    print(f"\n[{ts()}] Final validation...", flush=True)
    results = trainer.validate(lit, datamodule=dm)

    summary = os.path.join(out_dir, "results.txt")
    with open(summary, "w") as f:
        f.write(f"experiment  : {exp_name}\n")
        f.write(f"attn_type   : {ATTN_TYPE}\n")
        f.write(f"data_pct    : {DATA_PCT}\n")
        f.write(f"dataset     : {DATASET}\n")
        f.write(f"backbone    : {BACKBONE}\n")
        f.write(f"epochs      : {MAX_EPOCHS}\n")
        f.write(f"attn_params : {attn_params:,}\n")
        f.write("\n-- val metrics --\n")
        for d in results:
            for k, v in d.items():
                f.write(f"{k}: {v:.6f}\n")
    print(f"\n[{ts()}] Done! Results -> {summary}")
    return results


def run_all():
    """Run all 3 experiments (baseline, limix, mitra) and print a comparison table."""
    global ATTN_TYPE
    all_results = {}

    for attn in ["none", "limix", "mitra"]:
        ATTN_TYPE = attn
        print(f"\n{'#'*60}")
        print(f"#  Running: {attn}")
        print(f"{'#'*60}\n")
        try:
            res = run()
            metrics = {}
            for d in res:
                metrics.update(d)
            all_results[attn] = metrics
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {attn} failed: {e}")
            traceback.print_exc()
            all_results[attn] = {"error": str(e)}

        # Free GPU memory between runs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  RESULTS — {DATASET} @ {DATA_PCT}% data, {MAX_EPOCHS} epochs")
    print(f"{'='*60}")
    print(f"{'Attention':<12} {'mIoU':>10} {'Val Loss':>10} {'Attn Params':>12}")
    print(f"{'-'*50}")
    for attn in ["none", "limix", "mitra"]:
        m = all_results.get(attn, {})
        if "error" in m:
            print(f"{attn:<12} {'FAILED':>10}   ({m['error'][:40]})")
            continue
        iou = m.get("val/Multiclass_Jaccard_Index", 0)
        loss = m.get("val/loss", 0)
        exp_name = f"{DATASET}__{BACKBONE}__{attn}__pct{int(DATA_PCT)}"
        rfile = os.path.join(OUTPUT_DIR, exp_name, "results.txt")
        ap = "0"
        if os.path.isfile(rfile):
            for line in open(rfile):
                if "attn_params" in line:
                    ap = line.split(":")[1].strip()
        print(f"{attn:<12} {iou:>10.4f} {loss:>10.4f} {ap:>12}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all()
