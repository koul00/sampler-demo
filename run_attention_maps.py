"""
Run SAMPLER end-to-end: load slide representations, load the trained
likelihood-ratio model, and render tumor attention maps for two slides.

The large data files (hundreds of MB) are NOT kept in this repo. By default
this script reads them from ~/Downloads/SAMPLERtest, but you can point it
anywhere:

    python run_attention_maps.py --data-dir /path/to/SAMPLERtest

Outputs im1.png and im2.png into ./examples/.
"""
import os
import pickle
import argparse
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use("Agg")            # headless: save PNGs without opening a window
import matplotlib.pyplot as plt

import SAMPLER                    # the helper library (SAMPLER.py, same folder)

# ---------------------------------------------------------------- settings
fdim = 2048                                   # features per tile, per scale (InceptionV3)
pers = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]  # the 10 percentiles ("deciles")
dsr = 4                                        # downsample rate for the map
numclasses = 2                                 # tumor vs. normal


def load(data_dir, name):
    with open(os.path.join(data_dir, name), "rb") as f:
        return pickle.load(f)


def build_weights(Xtrain, Ytrain):
    """Feature importance: two-sample t-test between the two classes.
    Returns Wmat = -log(p-value) reshaped to (10 percentiles, 3*fdim)."""
    _, P = sp.stats.ttest_ind(Xtrain[Ytrain == 0, :], Xtrain[Ytrain == 1, :])
    rate = len(pers)
    Pmat = np.zeros((rate, 3 * fdim))
    for i in range(rate):
        Pmat[i, :fdim]           = P[i * fdim:(i + 1) * fdim]
        Pmat[i, fdim:2 * fdim]   = P[fdim + i * fdim:fdim + (i + 1) * fdim]
        Pmat[i, 2 * fdim:3 * fdim] = P[2 * fdim + i * fdim:2 * fdim + (i + 1) * fdim]
    return -np.log(Pmat), P


def make_map(data_dir, slide, Dtab, Wmat, cls, use_abs):
    """Score every tile of one slide and render its attention heatmap."""
    W = load(data_dir, slide + ".p")
    feats = W["feats"][::3, :]      # subsample tiles x3 (speed)
    Tlocs = W["locs"][::3, :]

    XR = feats[:, :fdim]            # small-tile features   ("R")
    XP = feats[:, fdim:2 * fdim]    # medium-tile features  ("P")
    XB = feats[:, 2 * fdim:]        # large-tile features   ("B")
    locs = (Tlocs / dsr).astype(np.int32)

    sR = SAMPLER.Sscore(XR, Dtab["XtabR"], Dtab[f"LLR_R_C_{cls}"], Wmat[:, :fdim], pers)
    sP = SAMPLER.Sscore(XP, Dtab["XtabP"], Dtab[f"LLR_P_C_{cls}"], Wmat[:, fdim:2 * fdim], pers)
    sB = SAMPLER.Sscore(XB, Dtab["XtabB"], Dtab[f"LLR_B_C_{cls}"], Wmat[:, 2 * fdim:], pers)

    scores = np.sum(sR, axis=1) + np.sum(sP, axis=1) + np.sum(sB, axis=1)
    if use_abs:
        scores = np.abs(scores)
    scores = SAMPLER.normalize_scores(scores)
    return SAMPLER.hmapgen3(scores, locs, int(512 / dsr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "examples"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[data] reading from {args.data_dir}")
    D = load(args.data_dir, "TCGA_BRCA_DX20X.p")
    Xtrain, Ytrain = D["Xtrain"], D["Ytrain"]
    print(f"[data] Xtrain={Xtrain.shape}  labels={np.bincount(Ytrain.astype(int))}")

    Wmat, _ = build_weights(Xtrain, Ytrain)
    Dtab = load(args.data_dir, "Dtab.p")
    print("[model] loaded Dtab.p (trained likelihood-ratio tables)")

    slides = [
        ("TCGA-AR-A0TP-01Z-00-DX1.58E6A3A6-D1D7-4AE5-87A5-E823F04CBB2F.svs", 0, False, "im1.png"),
        ("TCGA-B6-A0RQ-01Z-00-DX1.68B4F49D-9F81-4501-BC66-0349012077C8.svs", 1, True,  "im2.png"),
    ]
    for slide, cls, use_abs, out in slides:
        hmap = make_map(args.data_dir, slide, Dtab, Wmat, cls, use_abs)
        path = os.path.join(args.out_dir, out)
        plt.imsave(path, hmap)
        print(f"[map] class {cls}: wrote {path}  hmap={hmap.shape}")
    print("DONE")


if __name__ == "__main__":
    main()
