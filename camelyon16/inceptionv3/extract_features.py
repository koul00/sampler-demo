"""
extract_features.py

Faithful SAMPLER feature extraction on CAMELYON16 slides — the same recipe
the TCGA-BRCA demo's precomputed .p files were built with, run here on our
own 10% stratified CAMELYON16 cohort (16 normal, 11 tumor) instead:

- InceptionV3 (ImageNet-pretrained), GAP pooled to 2048-dim per tile
- 3 magnifications: 512 / 1536 / 2560 px tiles, 50% overlap
- SAMPLER.SAMPLERrep (10 percentiles per feature) applied per scale, then
  concatenated across scales -> 3 x 10 x 2048 = 61,440-dim per slide

~20x equivalent level (downsample 2.0 from the 40x base), background tiles
dropped via a cheap Otsu tissue mask computed once per slide at a low-res
level. ~20-100s/slide on Apple-MPS; ~18 min for all 27 slides.

Needs slides/ from https://github.com/koul00/lazyslide-wsi-pipeline (same
tumor_NNN.tif / normal_NNN.tif CAMELYON16 files used by that repo's cohort).

Output: cohort27_inceptionv3.npz — X (27, 61440), y (27,) tumor=1/normal=0,
names (27,). Run `run_decile.py` in this folder to evaluate it.
"""
import time
import numpy as np
import openslide
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

LEVEL = 1                      # ~20x equivalent (downsample 2.0 from the 40x base)
SCALES = [512, 1536, 2560]     # px, matching the SAMPLER paper's 3 magnifications
OVERLAP = 0.5
BATCH = 64
DECILES = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

SLIDES_DIR = "../../../lazyslide-wsi-pipeline/slides"  # sibling repo checkout
new27 = {
    "normal": [4, 16, 17, 21, 42, 54, 60, 62, 78, 84, 103, 108, 115, 116, 141, 150],
    "tumor": [5, 16, 23, 31, 38, 57, 58, 65, 75, 81, 93],
}
NAMES = [f"normal_{i:03d}" for i in new27["normal"]] + [f"tumor_{i:03d}" for i in new27["tumor"]]
LABELS = [0] * len(new27["normal"]) + [1] * len(new27["tumor"])


def load_model(device):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = nn.Identity()          # GAP output (2048-dim), skip the classifier head
    model.eval().to(device)
    return model, weights.transforms()


def tissue_mask(s, level_downsample):
    """Cheap Otsu tissue mask at a low-res level, scaled to LEVEL-pixel units."""
    tlevel = s.level_count - 3 if s.level_count >= 3 else s.level_count - 1
    tw, th = s.level_dimensions[tlevel]
    thumb = np.array(s.read_region((0, 0), tlevel, (tw, th)).convert("RGB"))
    sat = rgb2hsv(thumb)[:, :, 1]
    mask = ndi.binary_fill_holes(ndi.binary_opening(sat > threshold_otsu(sat), iterations=2))
    return mask, s.level_downsamples[tlevel] / level_downsample


def slide_vector(path, model, preprocess, device):
    s = openslide.OpenSlide(path)
    W, H = s.level_dimensions[LEVEL]
    ds = s.level_downsamples[LEVEL]
    tissue, t_scale = tissue_mask(s, ds)

    def has_tissue(x, y, tile):
        ty0, ty1 = int(y / t_scale), int((y + tile) / t_scale)
        tx0, tx1 = int(x / t_scale), int((x + tile) / t_scale)
        patch = tissue[max(0, ty0):ty1, max(0, tx0):tx1]
        return patch.size > 0 and patch.mean() > 0.3

    scale_vecs = []
    for scale in SCALES:
        stride = int(scale * (1 - OVERLAP))
        positions = [(x, y) for y in range(0, H - scale, stride)
                             for x in range(0, W - scale, stride)
                             if has_tissue(x, y, scale)]
        feats = []
        for i in range(0, len(positions), BATCH):
            batch_pos = positions[i:i + BATCH]
            imgs = [preprocess(s.read_region((int(x * ds), int(y * ds)), LEVEL, (scale, scale)).convert("RGB"))
                    for (x, y) in batch_pos]
            with torch.no_grad():
                out = model(torch.stack(imgs).to(device))
            feats.append(out.cpu().numpy())
        feats = np.concatenate(feats, axis=0) if feats else np.zeros((1, 2048))
        scale_vecs.append(np.ravel(np.percentile(feats, DECILES, axis=0)))
    return np.concatenate(scale_vecs)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  |  {len(NAMES)} slides", flush=True)
    model, preprocess = load_model(device)

    t0 = time.time()
    vecs = []
    for i, name in enumerate(NAMES, 1):
        ts = time.time()
        vecs.append(slide_vector(f"{SLIDES_DIR}/{name}.tif", model, preprocess, device))
        print(f"[{i}/{len(NAMES)}] {name} done  ({time.time()-ts:.1f}s, total {time.time()-t0:.1f}s)", flush=True)

    X = np.vstack(vecs)
    np.savez("cohort27_inceptionv3.npz", X=X, y=np.array(LABELS), names=np.array(NAMES))
    print(f"\nSAVED: X {X.shape}  (total {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
