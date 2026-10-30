# Experimental Setup and Methodology (presentation section 2, 20 marks)

Slide content and speaker notes for the methodology part of the video. The code is in
`notebooks/02_experimental_setup.ipynb` and `src/`. The numbers come from the Kaggle runs in
`results/runs/` (Tesla T4), and the slide charts are in `docs/figures/` (regenerate them with
`python docs/make_figures.py`).

Aim for about 2 minutes of speaking: 5 slides, roughly 25 seconds each.

---

## Slide 1: Tools and compute

- **Python 3, PyTorch 2** (mixed precision, AdamW), **segmentation-models-pytorch** for the pretrained encoder
- **OpenCV / pycocotools / scikit-image** for preprocessing and connected components
- **Kaggle Notebooks, NVIDIA T4 / P100 (16 GB)**, about 30 free GPU-hours a week, with the competition data attached directly. Notebooks 01 and 02 also run on Google Colab.
- Every run checkpoints each epoch and resumes in the next session. Training stops starting new epochs before Kaggle's 12-hour session limit, and each saved version's runs feed into the next.
- **GitHub** for code, with one notebook per pipeline stage (01 data → 02 training/evaluation → enhancement)

> *Notes:* "We trained on Kaggle's free GPUs, because Colab's free quota ran out after a few hours. Sessions end after 12 hours, so every run checkpoints each epoch and picks up where it left off in the next session, and all results go into one table that the rest of the team reads."

---

## Slide 2: Problem formulation

- The metric is **Panoptic Quality** = SQ × RQ: a predicted filament counts only if IoU > 0.5 with one ground-truth filament. Splitting one filament or merging two neighbours is penalised as much as missing one.
- Formulated as **bottom-up instance segmentation**: one U-Net predicts five dense maps, and instances are grouped from them afterwards.

| Output | Loss | Why |
|---|---|---|
| filament mask (`fg`) | BCE + Dice | which pixels are filament |
| instance boundary | BCE (pos. w. 3) + Dice | separates neighbours |
| contact between instances | BCE (pos. w. 10) + Dice | where to cut merged filaments |
| per-instance distance map | MSE | instance centres, used as watershed markers |
| filament spine | BCE (pos. w. 3) + Dice | keeps thin filaments continuous |

- **Why not Mask R-CNN?** Filaments are thin (median 11 px wide), long (up to about 1100 px) and curved, so their bounding boxes are mostly background and overlap heavily. A U-Net keeps full-resolution detail through its skip connections and runs on the whole 2048 × 2048 disk in one pass.

> *Notes:* "Because PQ scores instances, we predict both the filament mask and the cues needed to separate instances (boundaries, contacts, distance to the edge, spines) with one network. Ethan's section groups those maps into instances. Here we keep the grouping fixed and simple, connected components, so that model choices are compared fairly."

---

## Slide 3: Models and training algorithm

- **U-Net from scratch:** 6 levels (5 poolings), 32 → 512 channels, BatchNorm, 20.6 M parameters (79 MB fp32)
- **U-Net with an ImageNet ResNet-34 encoder:** 24.4 M parameters (93 MB fp32); the RGB first-layer filters are summed into one grey channel
- Output biases start at p = 0.01, because under 1% of pixels are filament (avoids a loss spike at the start)
- **Training:** 512 × 512 crops, 80% centred on a filament, random 90° rotations and flips; batch 8; **AdamW** (weight decay 1e-4), 1 epoch of warmup then cosine decay; fp16 AMP; gradient clipping at 1.0
- **Loss weighting:** pixels off the disk get zero weight and the weight ramps down near the limb (unreliable labels there). Each annotator's set is a separate sample, so annotator disagreement acts as label noise.

> *Notes:* "A 512 crop covers 95% of filament lengths with context to spare, and because the network is fully convolutional we can still run it on the full disk at test time. We compared a U-Net trained from scratch with one using a pretrained ResNet-34 encoder, because it is not obvious that ImageNet features transfer to single-channel H-alpha images."

---

## Slide 4: Validation and evaluation protocol

- **Split from notebook 01:** 600 / 107 observations in **7-day blocks**, stratified by instrument, so observations close in time never straddle the split and all annotation sets of an observation stay together
- Validation runs on the **full disk** (no crops). The prediction is scored against **every annotator's set**, averaged over sets, then over observations
- Metrics: **PQ** (mean ± std over observations, and pooled over all sets), SQ, RQ, pixel Dice, and **one-to-many / many-to-one** counts (fragmentation and merging)
- **Model selection on validation PQ** every 5 epochs, not on loss
- **Cost of the final model** (Tesla T4): 24.4 M parameters, 93 MB fp32 / 47 MB fp16, **61 min** to train (50 epochs, 74 s per epoch), **0.40 s per image** end to end (raw JPEG → disk fit → network → instance map)

> *Notes:* "We select checkpoints on PQ itself because the loss does not penalise merges or splits. Scoring against every annotator and then averaging per observation means observations with three annotators don't count three times."

---

## Slide 5: Tuning: staged search

| Stage | What is tested | Runs | Val PQ | Kept |
|---|---|---|---|---|
| A1 → A2 | fg only → all 5 outputs (U-Net from scratch) | 2 × 20 ep | 0.287 → **0.325** | 5 outputs |
| A3 | ImageNet ResNet-34 encoder, lr 3e-4 | 20 ep | **0.330** | ResNet-34 |
| B1 | lr 1e-4 / 3e-4 / 1e-3 | 2 × 20 ep (+ A3) | 0.327 / **0.330** / 0.324 | lr 3e-4 |
| B2 | soft consensus labels (α = 0.5); 768 crops (batch 4) | 2 × 20 ep | 0.341; **0.358** | 768 crops |
| C | ResNet-34, 5 outputs, lr 3e-4, 768 crops, 50 epochs, **3 seeds** | 3 × 50 ep | **0.380 ± 0.008** | (vs D) |
| D1–D3 | aimed at Stage C's errors: soft labels + 768 crops; 1024 crops (batch 2); 50% filament-centred crops | 3 × 20 ep | *pending Kaggle run* | |
| D | winner of B2/D1–D3, **80 epochs**, 3 seeds | 3 × 80 ep | *pending* | final model if better than C |
| post | fg threshold {0.3–0.7} × min area {0–200 px}, best seed | grid of 25 | 0.388 → **0.405** (thr 0.4, min area 200) | |

Charts: `docs/figures/stage_comparison.png` (this table as a bar chart), `val_pq_curves.png` (Stage A plus 768 crops), and `final_seeds.png` (the three final seeds).

What the search showed:
- **Noise level:** the three final seeds differ by ±0.008 PQ, so single-run differences smaller than about 0.01 are not meaningful. That covers the learning rate (all three within 0.006) and pretrained vs scratch (+0.005).
- **Clear gains:** the auxiliary outputs (+0.038; false positives 1614 → 1326 and fragmented filaments 216 → 189), 768 crops (+0.028; false positives 1347 → 966), and 50 epochs instead of 20 (+0.022). Soft labels gave +0.011, a borderline result.
- **ResNet-34 kept for speed:** the pretrained encoder scored about the same as the scratch U-Net but trained about 1.5× faster (33 s vs 50 s per epoch).
- **Main error:** false positives (about 850 vs 855 true positives) and fragmentation (about 127 filaments split in two or more, vs 27 merges), so RQ (0.57) is much lower than SQ (0.65). Removing blobs under 200 px gives +0.017, which suggests many false positives are small fragments.

- Each stage builds on the previous winner (greedy search), which kept Stages A–C to 10 runs, about 6 hours of training and validation on a Kaggle T4
- **Stage D** follows from analysing Stage C's errors, using training-side changes only (post-processing belongs to the Enhancement section). It combines soft labels with 768 crops, tries 1024 crops, and shows the network more filament-free crops to cut false positives. The winner is trained for 80 epochs, since Stage C was still improving at epoch 50. The final model is whichever of C and D scores higher, under the same fixed post-processing.
- Caveat to say aloud: the post-processing is tuned on the same validation set, so 0.405 is optimistic. The **Kaggle leaderboard** is the unbiased check.

> *Notes:* "We tuned greedily on a limited GPU budget. The three final seeds vary by about 0.008 PQ, which tells us which differences are real. Predicting the extra outputs, larger crops and longer training clearly helped. The learning rate barely mattered, and the pretrained encoder matched the scratch U-Net but trained one and a half times faster. The final model scores 0.380 ± 0.008 validation PQ, or 0.405 after tuning post-processing, and its main weakness is false positives and fragmented filaments, which is what our enhancement section tackles."

---

## Hand-offs

- **Results (section 3):** `results/runs/results.csv` (one row per run), `results/runs/<run>/val_per_set.csv` (per-image PQ, for best and worst cases), `history.csv`, and `results/runs/C_final_s1/test_pred/*.npz` (180 test instance maps, to encode into `submission.csv`).
  - Best validation observations (seed 1): `20210721123430_Ch` (PQ 0.74), `20110313082654_Uh` (0.68).
  - Worst: `20220418020950_Uh` and `20200105073250_Uh` (PQ 0; one or two filaments missed), and `20140818195814_Mh` (0.10; 10 false positives).
  - By instrument, PQ ranges from 0.36 (Mh, Bh) to 0.43 (Ch).
  - Annotators disagree: on the 39 validation observations with several annotation sets, the same prediction's PQ differs by 0.14 on average between annotators. That puts a ceiling on achievable PQ and is worth mentioning.
- **Enhancement (section 4):** `src.evaluate.collect_probs` caches all five output maps for the validation set. Pass your own `postproc(probs, disk, **kw) -> instance map` to `grid_search` / `score_collected` to compare against the connected-components baseline under the same protocol. For compression, `model_size` and `benchmark` give the before/after numbers.
  - Start from the errors: false positives and fragmentation, not merges.
  - The best minimum area (200 px) was at the edge of the grid, so try larger values too.
