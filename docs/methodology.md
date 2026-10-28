# Experimental Setup and Methodology (presentation section 2, 20 marks)

Slide content and speaker notes for the methodology part of the video. The code is in
`notebooks/02_experimental_setup.ipynb` and `src/`. Numbers marked **[fill in]** come from
`runs/results.csv` on the shared drive once the experiments have run.

Aim for about 2 minutes of speaking: 5 slides, roughly 25 seconds each.

---

## Slide 1: Tools and compute

- **Python 3, PyTorch 2** (mixed precision, AdamW), **segmentation-models-pytorch** for the pretrained encoder
- **OpenCV / pycocotools / scikit-image** for preprocessing and connected components
- **Google Colab, NVIDIA T4 (16 GB)**. Every run logs its config, history, checkpoint and scores to the shared Drive and resumes after a disconnect
- **GitHub** for code, with one notebook per pipeline stage (01 data → 02 training/evaluation → enhancement)

> *Notes:* "Everything runs on a free Colab T4. Because sessions drop, every run checkpoints each epoch to Drive and picks up where it left off, and all results go into one table that the rest of the team reads."

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
- Cost reported: parameters, size, training minutes, and **end-to-end** seconds per image (raw JPEG → disk fit → network → instance map)

> *Notes:* "We select checkpoints on PQ itself because the loss does not penalise merges or splits. Scoring against every annotator and then averaging per observation means observations with three annotators don't count three times."

---

## Slide 5: Tuning: staged search

| Stage | What is tested | Runs | Result (val PQ) |
|---|---|---|---|
| A1 / A2 | fg-only vs multi-task outputs (U-Net from scratch) | 2 × 20 ep | [fill in] / [fill in] |
| A3 | ImageNet ResNet-34 encoder (multi-task) | 20 ep | [fill in] |
| B1 | learning rate {1e-4, 3e-4, 1e-3} on the A winner | 2–3 × 20 ep | [fill in] |
| B2 | soft consensus labels (α = 0.5); 768 crops | 2 × 20 ep | [fill in] |
| C | best config, 50 epochs, **3 seeds** | 3 × 50 ep | **[mean ± std]** |
| post | fg threshold {0.3–0.7} × min area {0–200 px}, grid search on cached outputs | — | [fill in] |

- Each stage builds on the previous winner (greedy search), which keeps the GPU budget to roughly 10 runs
- Plot for the slide: `plot_history("A1_unet_fg", "A2_unet_mt", "A3_r34_mt")` (training loss and validation PQ curves)
- Caveat to say aloud: the threshold is tuned on the same validation set, so that number is optimistic. The **Kaggle leaderboard** is the unbiased check.

> *Notes:* "With a limited GPU budget we tuned greedily: architecture and formulation first, then learning rate, then label handling and crop size, and finally three seeds of the winner to report a mean and standard deviation."

---

## Hand-offs

- **Results (section 3):** `runs/results.csv` (one row per run), `runs/<run>/val_per_set.csv` (per-image PQ, for best and worst cases), `runs/<run>/history.csv`, and `runs/C_final_s*/test_pred/*.npz` (test instance maps, to encode into `submission.csv`)
- **Enhancement (section 4):** `src.evaluate.collect_probs` caches all five output maps for the validation set. Pass your own `postproc(probs, disk, **kw) -> instance map` to `grid_search` / `score_collected` to compare against the connected-components baseline under the same protocol. For compression, `model_size` and `benchmark` give the before/after numbers.
