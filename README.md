# SAMPLER demo — tumor attention maps on whole-slide images


A small, runnable walkthrough of **SAMPLER** (Foroughi Pour et al., *eBioMedicine*
2023), a method that turns a gigapixel pathology slide into a compact fixed-size
vector, trains a light classifier on those vectors, and paints **attention maps**
showing which regions of a slide look tumor-like.

**Where this sits in the timeline:** the SAMPLER vector is built from *learned*
deep features (InceptionV3) instead of the hand-crafted texture/shape features of the
classical era — but it is still fed to **classical classifiers** (Logistic Regression,
SVM, Random Forest). It's the bridge between hand-crafted-feature pathology and
today's foundation models.

This repo bundles the SAMPLER helper library, a clean driver script, the original
example notebook, and two example output maps. The large data files are downloaded
separately (see [`data/README.md`](data/README.md)).

![example attention map](examples/im1.png)

## Quick start

```bash
pip install -r requirements.txt
# download the data bundle into data/, then either:
jupyter notebook demo.ipynb          # run top-to-bottom: attention maps + classifier
# ...or just the attention-map driver:
python run_attention_maps.py --data-dir /path/to/SAMPLERtest
# -> writes examples/im1.png and examples/im2.png
```

## Classical classifier on the SAMPLER representation

`demo.ipynb` Section 7 trains off-the-shelf **classical classifiers** on the 61,440-long
SAMPLER vectors and predicts **tumor vs. normal** on held-out TCGA-BRCA slides
(693 train / 298 test, ~20% tumor). Same classifiers the classical era relied on —
now standing on deep features:

| Classifier | Test AUC | Test Accuracy |
|-----------|:--------:|:-------------:|
| Logistic Regression | **0.917** | 0.889 |
| SVM (RBF) | 0.910 | 0.893 |
| Random Forest | 0.868 | 0.836 |

## What SAMPLER actually does

A whole-slide image (WSI) is enormous (~100,000 × 100,000 pixels), so it is first
cut into thousands of small **tiles**. Each tile is passed through a pretrained
InceptionV3 network, giving every tile a **2048-number feature vector** at 3
magnifications (small / medium / large regions — nicknamed **R / P / B** in the
code). So one slide = a big table of `(number_of_tiles) × 6144` numbers, and every
slide has a *different* number of tiles. Classifiers need fixed-size inputs, so:

### Step 1 — SAMPLER representation (the core trick)
For each of the 2048 features, take **10 percentiles** across all the slide's tiles
(the 5th, 15th, … 95th). This summarizes the *distribution* of each feature over the
slide, independent of how many tiles it has.

```
one slide:  (n_tiles × 2048)  →  10 percentiles × 2048 = 20,480 numbers
× 3 magnifications                                   → 61,440 numbers
```

That 61,440-long vector is the slide's SAMPLER representation. It's the one line the
README of the original repo highlights: `np.ravel(np.percentile(X, pers, axis=0))`
(`SAMPLER.SAMPLERrep`).

In this demo the representations are precomputed: `TCGA_BRCA_DX20X.p` holds
`Xtrain (693, 61440)` and `Xtest (298, 61440)` with binary labels (tumor vs normal;
train split is 551 / 142).

### Step 2 — Which features matter (weights)
A two-sample **t-test** compares each feature between the two classes. Small p-value
= discriminative feature. The weight used later is `Wmat = -log(p)`, so strongly
class-separating features count more. (`build_weights` in the driver.)

### Step 3 — Train the model (likelihood-ratio tables)
For every (feature, percentile) slot, SAMPLER estimates the value distribution in
class 0 vs class 1 using a kernel density estimate, then stores the **log-likelihood
ratio** on a grid. That's `Dtab.p`:

```
XtabR / LLR_R_C_0 / LLR_R_C_1 …   each shaped (2048 features, 10 percentiles, 20 grid points)
```

Given any tile's feature value, the model looks up "how much more class-0-like vs
class-1-like is this?" (`SAMPLER.Fscore` / `SAMPLER.Sscore` do the interpolation).
This is unusual — no neural-net training, just per-feature density lookups — which is
why it's fast.

### Step 4 — Score every tile and paint the map
For a target slide, each tile is scored at all 3 magnifications, the scores are
summed, normalized to 0–1, and stamped back onto the slide's tile grid using each
tile's `locs` (top-left pixel coordinate). `SAMPLER.hmapgen3` builds an RGB image:

- **red**  = high score (looks like the target class / tumor)
- **blue** = low score
- **white** = no tissue

The two example slides have ~10,000 tiles each; the resulting maps are
`~10368 × 9152` and `~7616 × 14528` pixels.

## Files in this repo

| File | Role |
|------|------|
| `SAMPLER.py` | The helper library (percentiles, scoring, map generation) |
| `run_attention_maps.py` | Clean end-to-end driver (this is the thing to run) |
| `attention_maps.ipynb` | Original annotated notebook from the authors' test bundle |
| `examples/im1.png`, `im2.png` | Example output maps |
| `data/` | Where the downloaded big files go (git-ignored) |
| `ORIGINAL_README.md` | The upstream project's README (download links, citation) |

## Shape cheat-sheet

| Object | Shape | Meaning |
|--------|-------|---------|
| tile features (per slide) | `(n_tiles, 6144)` | 2048 × 3 magnifications |
| SAMPLER representation | `(61440,)` | 2048 × 10 percentiles × 3 |
| `Xtrain` / `Xtest` | `(693, 61440)` / `(298, 61440)` | one row per slide |
| `Dtab` LLR tables | `(2048, 10, 20)` | feature × percentile × grid |
| attention map | `(H, W, 3)` | RGB image over the tile grid |

## Credit

Method and data: **TheJacksonLaboratory/SAMPLER**
(https://github.com/TheJacksonLaboratory/SAMPLER). This repo is a personal,
runnable demo/notes layer on top of their released code and test bundle.
