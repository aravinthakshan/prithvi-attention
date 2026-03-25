"""
Colab / Kaggle runner — edit the CONFIG BLOCK below and run this file.

Supports 3 task types across 7 datasets:
  Segmentation : firescars, burnintensity, sen1floods11, landslide, multicrop
  Classification: sen4map_crops, sen4map_landcover   (requires local HDF5 files)
  Regression   : biomassters

    !python colab_run.py --attn orion --dataset firescars
    !python colab_run.py --attn limix --dataset biomassters
    !python colab_run.py                          # runs all 4 attn types on default dataset
"""

# ============================================================
#  CONFIG BLOCK — edit these for each experiment
# ============================================================

DATASET    = "firescars"           # see CFGS keys below
ATTN_TYPE  = "none"                # "none" | "limix" | "mitra" | "orion"
DATA_PCT   = 100.0                 # % of training data
BACKBONE   = "prithvi_eo_v2_300_tl"

DATA_ROOT  = None                  # None = auto-download  |  "/your/path" = skip download
OUTPUT_DIR = "./outputs"
HF_TOKEN   = "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"

MAX_EPOCHS   = 2
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

# ============================================================
# Dataset download / extract helpers
# ============================================================

HF_DATASET_IDS = {
    "firescars":     "ibm-nasa-geospatial/hls_burn_scars",
    "burnintensity": "ibm-nasa-geospatial/burn_intensity",
    "sen1floods11":  "ibm-nasa-geospatial/Sen1Floods11",
    "landslide":     "ibm-nasa-geospatial/Landslide4sense",
    "multicrop":     "ibm-nasa-geospatial/multi-temporal-crop-classification",
    "biomassters":   "ibm-nasa-geospatial/BioMassters",
}

def get_data_root(dataset, data_root, output_dir):
    from huggingface_hub import snapshot_download
    if data_root and os.path.isdir(data_root):
        _extract(data_root)
        return _resolve(data_root)
    hf_id = HF_DATASET_IDS.get(dataset)
    if hf_id is None:
        raise ValueError(f"No HF dataset ID for '{dataset}'. Set DATA_ROOT manually.")
    dest = os.path.join(output_dir, "data", dataset)
    if not (os.path.isdir(dest) and os.listdir(dest)):
        print(f"[{ts()}] Downloading {hf_id}...", flush=True)
        os.makedirs(dest, exist_ok=True)
        snapshot_download(repo_id=hf_id, repo_type="dataset",
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

# ============================================================
# Dataset configs — task type, bands, classes, metrics, loss
# ============================================================

TASK_SEGMENTATION = "segmentation"
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"

CFGS = {
    # ---- Segmentation ----
    "firescars": dict(
        task=TASK_SEGMENTATION,
        num_classes=2, num_frames=1, loss="ce", ignore_index=-1,
        decoder="UperNetDecoder", decoder_channels=256,
        bands=["BLUE","GREEN","RED","NIR_BROAD","SWIR_1","SWIR_2"],
        coords=["time","location"],
        monitor="val/mIoU", mode="max",
    ),
    "burnintensity": dict(
        task=TASK_SEGMENTATION,
        num_classes=5, num_frames=3, loss="ce", ignore_index=-1,
        decoder="UperNetDecoder", decoder_channels=512,
        bands=["BLUE","GREEN","RED","NIR","SWIR_1","SWIR_2"],
        coords=["location"],
        monitor="val/mIoU", mode="max",
        class_weights=[0.018, 0.14, 0.076, 0.11, 0.65],
    ),
    "sen1floods11": dict(
        task=TASK_SEGMENTATION,
        num_classes=2, num_frames=1, loss="ce", ignore_index=-1,
        decoder="UperNetDecoder", decoder_channels=256,
        bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
        coords=[],
        monitor="val/mIoU", mode="max",
    ),
    "landslide": dict(
        task=TASK_SEGMENTATION,
        num_classes=2, num_frames=1, loss="focal", ignore_index=-1,
        decoder="UperNetDecoder", decoder_channels=256,
        bands=["BLUE","GREEN","RED","NIR_BROAD","SWIR_1","SWIR_2"],
        coords=[],
        monitor="val/mIoU", mode="max",
    ),
    "multicrop": dict(
        task=TASK_SEGMENTATION,
        num_classes=13, num_frames=3, loss="ce", ignore_index=-1,
        decoder="UperNetDecoder", decoder_channels=256,
        bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
        coords=["time","location"],
        monitor="val/mIoU", mode="max",
        class_weights=[0.386, 0.661, 0.548, 0.640, 0.877, 0.925,
                       3.249, 1.542, 2.175, 2.272, 3.063, 3.626, 1.199],
    ),
    # ---- Classification ----
    "sen4map_crops": dict(
        task=TASK_CLASSIFICATION,
        num_classes=8, num_frames=12, loss="ce", ignore_index=-1,
        decoder="IdentityDecoder", decoder_channels=None,
        bands=["BLUE","GREEN","RED","RED_EDGE_1","RED_EDGE_2","RED_EDGE_3",
               "NIR_BROAD","NIR_NARROW","SWIR_1","SWIR_2"],
        coords=[],
        monitor="val/Overall_Accuracy", mode="max",
    ),
    "sen4map_landcover": dict(
        task=TASK_CLASSIFICATION,
        num_classes=10, num_frames=12, loss="ce", ignore_index=-1,
        decoder="IdentityDecoder", decoder_channels=None,
        bands=["BLUE","GREEN","RED","RED_EDGE_1","RED_EDGE_2","RED_EDGE_3",
               "NIR_BROAD","NIR_NARROW","SWIR_1","SWIR_2"],
        coords=[],
        monitor="val/Overall_Accuracy", mode="max",
    ),
    # ---- Regression ----
    "biomassters": dict(
        task=TASK_REGRESSION,
        num_classes=1, num_frames=4, loss="mse", ignore_index=-1,
        decoder="UperNetDecoder", decoder_channels=512,
        bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
        coords=[],
        monitor="val/loss", mode="min",
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

# ============================================================
# Lightning modules — Segmentation, Classification, Regression
# ============================================================

import lightning.pytorch as pl
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy


class LitSeg(pl.LightningModule):
    """Semantic segmentation with mIoU tracking."""
    def __init__(self, model, num_classes, ignore_index, lr, loss_type="ce", class_weights=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.loss_type = loss_type
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index)

    def _get_logits(self, out):
        if hasattr(out, 'output'): return out.output
        if isinstance(out, (list, tuple)): return out[0]
        return out

    def _step(self, batch, phase):
        img = batch["image"]
        mask = batch["mask"]
        if img.ndim == 4: img = img.unsqueeze(2)
        if mask.ndim == 4: mask = mask.squeeze(1)
        logits = self._get_logits(self.model(img))

        if self.loss_type == "focal":
            loss = self._focal_loss(logits, mask.long())
        else:
            loss = F.cross_entropy(logits, mask.long(), weight=self.class_weights,
                                   ignore_index=self.ignore_index)
        if phase == "val":
            self.val_iou.update(logits.argmax(1), mask.long())
        self.log(f"{phase}/loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def _focal_loss(self, logits, targets, gamma=2.0, alpha=0.25):
        ce = F.cross_entropy(logits, targets, weight=self.class_weights,
                             ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce)
        return (alpha * (1 - pt) ** gamma * ce).mean()

    def training_step(self, batch, batch_idx): return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def on_validation_epoch_end(self):
        self.log("val/mIoU", self.val_iou.compute(), prog_bar=False)
        self.val_iou.reset()
    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


class LitCls(pl.LightningModule):
    """Image classification with accuracy tracking."""
    def __init__(self, model, num_classes, lr, class_weights=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

    def _get_logits(self, out):
        if hasattr(out, 'output'): return out.output
        if isinstance(out, (list, tuple)): return out[0]
        return out

    def _step(self, batch, phase):
        img = batch["image"]
        label = batch["label"]
        if img.ndim == 4: img = img.unsqueeze(2)
        logits = self._get_logits(self.model(img))  # (B, num_classes)
        if logits.ndim > 2:
            logits = logits.mean(dim=list(range(2, logits.ndim)))
        loss = F.cross_entropy(logits, label.long(), weight=self.class_weights)
        if phase == "val":
            self.val_acc.update(logits.argmax(1), label.long())
        self.log(f"{phase}/loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def on_validation_epoch_end(self):
        self.log("val/Overall_Accuracy", self.val_acc.compute(), prog_bar=False)
        self.val_acc.reset()
    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


class LitReg(pl.LightningModule):
    """Pixelwise regression with RMSE tracking."""
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self._val_se_sum = 0.0
        self._val_n = 0

    def _get_logits(self, out):
        if hasattr(out, 'output'): return out.output
        if isinstance(out, (list, tuple)): return out[0]
        return out

    def _step(self, batch, phase):
        img = batch["image"]
        target = batch["mask"]
        if img.ndim == 4: img = img.unsqueeze(2)
        if target.ndim == 4: target = target.squeeze(1)
        preds = self._get_logits(self.model(img))
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        loss = F.mse_loss(preds, target.float())
        if phase == "val":
            self._val_se_sum += (preds - target.float()).pow(2).sum().item()
            self._val_n += target.numel()
        self.log(f"{phase}/loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def on_validation_epoch_end(self):
        rmse = (self._val_se_sum / max(self._val_n, 1)) ** 0.5
        self.log("val/RMSE", rmse, prog_bar=False)
        self._val_se_sum = 0.0; self._val_n = 0
    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ============================================================
# DataModule builder — dispatches to the right TerraTorch DM
# ============================================================

def make_datamodule(dataset, data_root, batch_size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    nw = 2  # Colab-safe

    if dataset == "firescars":
        from terratorch.datamodules import FireScarsNonGeoDataModule
        return FireScarsNonGeoDataModule(
            data_root=data_root, batch_size=batch_size, num_workers=nw,
            no_data_replace=0, no_label_replace=-1, use_metadata=True,
            train_transform=A.Compose([A.Resize(224,224), A.HorizontalFlip(0.5),
                                       A.VerticalFlip(0.5), ToTensorV2()]),
            val_transform=A.Compose([A.Resize(224,224), ToTensorV2()]),
            test_transform=A.Compose([A.Resize(224,224), ToTensorV2()]),
        )

    elif dataset == "burnintensity":
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

    elif dataset == "sen1floods11":
        from terratorch.datamodules import Sen1Floods11NonGeoDataModule
        return Sen1Floods11NonGeoDataModule(
            data_root=data_root, batch_size=batch_size, num_workers=nw,
            constant_scale=0.0001, no_data_replace=0, no_label_replace=-1,
            use_metadata=False,
            bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
            train_transform=A.Compose([A.Resize(224,224), A.HorizontalFlip(0.5),
                                       A.VerticalFlip(0.5), ToTensorV2()]),
            val_transform=A.Compose([A.Resize(224,224), ToTensorV2()]),
            test_transform=A.Compose([A.Resize(224,224), ToTensorV2()]),
        )

    elif dataset == "landslide":
        from terratorch.datamodules import Landslide4SenseNonGeoDataModule
        return Landslide4SenseNonGeoDataModule(
            data_root=data_root, batch_size=batch_size, num_workers=nw,
            bands=["BLUE","GREEN","RED","NIR_BROAD","SWIR_1","SWIR_2"],
            train_transform=A.Compose([A.Resize(224,224), A.Flip(), ToTensorV2()]),
            val_transform=A.Compose([A.Resize(224,224), ToTensorV2()]),
            test_transform=A.Compose([A.Resize(224,224), ToTensorV2()]),
        )

    elif dataset == "multicrop":
        from terratorch.datamodules import MultiTemporalCropClassificationDataModule
        from terratorch.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels
        return MultiTemporalCropClassificationDataModule(
            data_root=data_root, batch_size=batch_size, num_workers=nw,
            reduce_zero_label=True, expand_temporal_dimension=True, use_metadata=True,
            bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
            train_transform=A.Compose([FlattenTemporalIntoChannels(), A.Flip(), ToTensorV2(),
                                       UnflattenTemporalFromChannels(n_timesteps=3)]),
            val_transform=A.Compose([FlattenTemporalIntoChannels(), A.Flip(), ToTensorV2(),
                                     UnflattenTemporalFromChannels(n_timesteps=3)]),
        )

    elif dataset in ("sen4map_crops", "sen4map_landcover"):
        from terratorch.datamodules import Sen4MapLucasDataModule
        if not data_root:
            raise ValueError(f"Sen4Map requires local HDF5 files. Set DATA_ROOT to the directory "
                             f"containing train.h5, val.h5, test.h5 and their *_keys.pkl files.")
        cls_map = "crops" if dataset == "sen4map_crops" else None
        pkl_suffix = "_crop_keys.pkl" if dataset == "sen4map_crops" else "_keys.pkl"
        return Sen4MapLucasDataModule(
            batch_size=batch_size, num_workers=nw,
            train_hdf5_path=os.path.join(data_root, "train.h5"),
            train_hdf5_keys_path=os.path.join(data_root, f"train{pkl_suffix}"),
            val_hdf5_path=os.path.join(data_root, "val.h5"),
            val_hdf5_keys_path=os.path.join(data_root, f"val{pkl_suffix}"),
            test_hdf5_path=os.path.join(data_root, "test.h5"),
            test_hdf5_keys_path=os.path.join(data_root, f"test{pkl_suffix}"),
            dataset_bands=["BLUE","GREEN","RED","RED_EDGE_1","RED_EDGE_2","RED_EDGE_3",
                           "NIR_BROAD","NIR_NARROW","SWIR_1","SWIR_2"],
            input_bands=["BLUE","GREEN","RED","RED_EDGE_1","RED_EDGE_2","RED_EDGE_3",
                         "NIR_BROAD","NIR_NARROW","SWIR_1","SWIR_2"],
            crop_size=15, train_shuffle=True, resize=True,
            resize_to=[224, 224], resize_interpolation="bilinear",
            **({"classification_map": cls_map} if cls_map else {}),
        )

    elif dataset == "biomassters":
        from terratorch.datamodules import BioMasstersNonGeoDataModule
        from terratorch.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels
        return BioMasstersNonGeoDataModule(
            data_root=data_root, batch_size=batch_size, num_workers=nw,
            bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
            sensors=["S2"], as_time_series=True, use_four_frames=True,
            train_transform=A.Compose([FlattenTemporalIntoChannels(), A.Resize(224,224),
                                       A.Flip(), ToTensorV2(),
                                       UnflattenTemporalFromChannels(n_timesteps=4)]),
            val_transform=A.Compose([FlattenTemporalIntoChannels(), A.Resize(224,224),
                                     A.Flip(), ToTensorV2(),
                                     UnflattenTemporalFromChannels(n_timesteps=4)]),
        )

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(CFGS.keys())}")


# ============================================================
# Helpers
# ============================================================

def subset_dataset(ds, pct, seed):
    n = len(ds)
    k = max(1, min(n, math.ceil(n * pct / 100.0)))
    idx = sorted(random.Random(seed).sample(range(n), k))
    return Subset(ds, idx)


def wrap_patch_embed(pe, ca):
    """Wrap patch_embed to run channel_attn after it, proxying original attributes."""
    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.pe = pe
            self.ca = ca
        def forward(self, x, *a, **kw):
            return self.ca(self.pe(x, *a, **kw))
        def __getattr__(self, name):
            try: return super().__getattr__(name)
            except AttributeError: return getattr(self.pe, name)
    return W()


def build_lit_module(cfg, model):
    """Build the right Lightning module for the task type."""
    task = cfg["task"]
    if task == TASK_SEGMENTATION:
        return LitSeg(model=model, num_classes=cfg["num_classes"],
                      ignore_index=cfg["ignore_index"], lr=LR,
                      loss_type=cfg["loss"], class_weights=cfg.get("class_weights"))
    elif task == TASK_CLASSIFICATION:
        return LitCls(model=model, num_classes=cfg["num_classes"], lr=LR,
                      class_weights=cfg.get("class_weights"))
    elif task == TASK_REGRESSION:
        return LitReg(model=model, lr=LR)
    else:
        raise ValueError(f"Unknown task type: {task}")


def get_metric_name(cfg):
    """Return the primary metric name for display."""
    task = cfg["task"]
    if task == TASK_SEGMENTATION: return "mIoU"
    if task == TASK_CLASSIFICATION: return "Accuracy"
    if task == TASK_REGRESSION: return "RMSE"
    return "loss"


def get_metric_value(metrics, cfg):
    """Extract the primary metric value from trainer metrics dict."""
    return metrics.get(cfg["monitor"], 0)


# ============================================================
# Main run
# ============================================================

def run():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    pl.seed_everything(SEED, workers=True)

    cfg = CFGS[DATASET]
    task_type = cfg["task"]
    embed_dim = EMBED_DIM[BACKBONE]
    neck_idx = NECK_IDX[BACKBONE]
    n_bands = len(cfg["bands"])

    exp_name = f"{DATASET}__{BACKBONE}__{ATTN_TYPE}__pct{int(DATA_PCT)}"
    out_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Data root
    data_root = get_data_root(DATASET, DATA_ROOT, OUTPUT_DIR)
    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Task       : {task_type}")
    print(f"  Attention  : {ATTN_TYPE}")
    print(f"  Data %     : {DATA_PCT}%")
    print(f"  Backbone   : {BACKBONE}  (D={embed_dim})")
    print(f"  Dataset    : {DATASET}  T={cfg['num_frames']}, {cfg['num_classes']} classes/outputs")
    print(f"{'='*60}\n")

    # 2. Build model via terratorch factory
    print(f"[{ts()}] Building model...", flush=True)
    from terratorch.datasets import HLSBands
    is_tl = BACKBONE.endswith("_tl")

    if task_type in (TASK_SEGMENTATION, TASK_REGRESSION):
        from terratorch.models import EncoderDecoderFactory
        band_enums = [getattr(HLSBands, b) for b in cfg["bands"]]

        factory_task = "segmentation" if task_type == TASK_SEGMENTATION else "regression"
        num_out = cfg["num_classes"]

        model_args = dict(
            backbone=BACKBONE, backbone_pretrained=True,
            backbone_bands=band_enums,
            backbone_coords_encoding=cfg["coords"] if is_tl else [],
            decoder=cfg["decoder"], decoder_channels=cfg["decoder_channels"],
            decoder_scale_modules=True, num_classes=num_out,
            rescale=True, head_dropout=0.1,
            necks=[{"name": "SelectIndices", "indices": neck_idx},
                   {"name": "ReshapeTokensToImage",
                    **({"effective_time_dim": cfg["num_frames"]} if cfg["num_frames"] > 1 else {})}],
        )
        if cfg["num_frames"] > 1:
            model_args["backbone_num_frames"] = cfg["num_frames"]

        model = EncoderDecoderFactory().build_model(task=factory_task, **model_args)

    elif task_type == TASK_CLASSIFICATION:
        from terratorch.models import PrithviModelFactory
        band_enums = [getattr(HLSBands, b) for b in cfg["bands"]]

        model_args = dict(
            backbone=BACKBONE, pretrained=True,
            bands=band_enums,
            decoder="IdentityDecoder",
            head_dim_list=[384, 128],
            in_channels=n_bands,
            num_frames=cfg["num_frames"],
            num_classes=cfg["num_classes"],
            head_dropout=0.1,
            backbone_patch_size=16,
            backbone_pretrain_img_size=224,
        )
        model = PrithviModelFactory().build_model(task="classification", **model_args)

    print(f"[{ts()}] Model built. Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # 3. Channel attention
    ch_attn = build_channel_attention(ATTN_TYPE, embed_dim, n_bands, ATTN_HEADS)
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

    # 4. Lightning module
    print(f"[{ts()}] Building Lightning module...", flush=True)
    lit = build_lit_module(cfg, model)
    print(f"[{ts()}] Lightning module ready.", flush=True)

    # 5. DataModule
    print(f"[{ts()}] Setting up data...", flush=True)
    dm = make_datamodule(DATASET, data_root, BATCH_SIZE)
    dm.setup("fit")

    train_ds_attr = None
    for attr in ("train_dataset", "dataset_train", "train_ds"):
        if hasattr(dm, attr) and getattr(dm, attr) is not None:
            train_ds_attr = attr; break
    if train_ds_attr:
        print(f"[{ts()}] Train samples: {len(getattr(dm, train_ds_attr))}", flush=True)

    if DATA_PCT < 100.0 and train_ds_attr:
        orig = getattr(dm, train_ds_attr)
        setattr(dm, train_ds_attr, subset_dataset(orig, DATA_PCT, SEED))
        print(f"[{ts()}] Subset: {len(getattr(dm, train_ds_attr))}/{len(orig)} ({DATA_PCT}%)", flush=True)

    # 6. Quick sanity forward pass
    print(f"[{ts()}] Sanity check: loading 1 batch...", flush=True)
    if train_ds_attr:
        dl = DataLoader(getattr(dm, train_ds_attr),
                        batch_size=min(4, len(getattr(dm, train_ds_attr))),
                        shuffle=False, num_workers=0)
    else:
        dl = dm.train_dataloader()
    batch = next(iter(dl))
    img_key = "image"
    target_key = "mask" if task_type != TASK_CLASSIFICATION else "label"
    print(f"[{ts()}] Batch loaded — {img_key}: {batch[img_key].shape}, "
          f"{target_key}: {batch[target_key].shape}", flush=True)

    print(f"[{ts()}] Sanity check: forward pass...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    with torch.no_grad():
        img = batch[img_key].to(device)
        if img.ndim == 4: img = img.unsqueeze(2)
        print(f"[{ts()}] Forward input shape: {img.shape}", flush=True)
        out = model(img)
    if hasattr(out, 'output'):
        print(f"[{ts()}] Forward OK — output: {out.output.shape}", flush=True)
    elif isinstance(out, (list, tuple)):
        print(f"[{ts()}] Forward OK — output: {out[0].shape}", flush=True)
    else:
        print(f"[{ts()}] Forward OK — output: {out.shape}", flush=True)
    model.cpu(); model.train()

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

    # 9. Final validation
    print(f"\n[{ts()}] Final validation...", flush=True)
    results = trainer.validate(lit, datamodule=dm)

    summary = os.path.join(out_dir, "results.txt")
    with open(summary, "w") as f:
        f.write(f"experiment  : {exp_name}\n")
        f.write(f"task        : {task_type}\n")
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


# ============================================================
# Run all attention types and print comparison
# ============================================================

def run_all():
    global ATTN_TYPE
    all_results = {}
    cfg = CFGS[DATASET]

    for attn in ["none", "limix", "mitra", "orion"]:
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
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    metric_name = get_metric_name(cfg)
    print(f"\n{'='*60}")
    print(f"  RESULTS — {DATASET} ({cfg['task']}) @ {DATA_PCT}% data, {MAX_EPOCHS} epochs")
    print(f"{'='*60}")
    print(f"{'Attention':<12} {metric_name:>10} {'Val Loss':>10} {'Attn Params':>12}")
    print(f"{'-'*50}")
    for attn in ["none", "limix", "mitra", "orion"]:
        m = all_results.get(attn, {})
        if "error" in m:
            print(f"{attn:<12} {'FAILED':>10}   ({m['error'][:40]})")
            continue
        primary = get_metric_value(m, cfg)
        loss = m.get("val/loss", 0)
        exp_name = f"{DATASET}__{BACKBONE}__{attn}__pct{int(DATA_PCT)}"
        rfile = os.path.join(OUTPUT_DIR, exp_name, "results.txt")
        ap = "0"
        if os.path.isfile(rfile):
            for line in open(rfile):
                if "attn_params" in line:
                    ap = line.split(":")[1].strip()
        print(f"{attn:<12} {primary:>10.4f} {loss:>10.4f} {ap:>12}")
    print(f"{'='*60}\n")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prithvi + Channel Attention experiments")
    p.add_argument("--dataset", default=None,
                   help=f"Dataset: {', '.join(CFGS.keys())} (default: {DATASET})")
    p.add_argument("--data_pct", type=float, default=None,
                   help="% of training data (default: 100)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Max epochs (default: 2)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size (default: 8)")
    p.add_argument("--backbone", default=None,
                   help="Backbone name (default: prithvi_eo_v2_300_tl)")
    p.add_argument("--attn", default=None,
                   help="Single attention type: none|limix|mitra|orion (default: run all)")
    p.add_argument("--data_root", default=None,
                   help="Manual data root path (default: auto-download)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 5e-5)")
    args, _ = p.parse_known_args()

    if args.dataset:    DATASET = args.dataset
    if args.data_pct is not None: DATA_PCT = args.data_pct
    if args.epochs:     MAX_EPOCHS = args.epochs
    if args.batch_size: BATCH_SIZE = args.batch_size
    if args.backbone:   BACKBONE = args.backbone
    if args.data_root:  DATA_ROOT = args.data_root
    if args.lr:         LR = args.lr

    if args.attn:
        ATTN_TYPE = args.attn
        run()
    else:
        run_all()
