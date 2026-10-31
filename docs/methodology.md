# Experimental Setup and Methodology (presentation section 2, 20 marks)

Slide content and speaker notes for the methodology part of the video. The code is in
`notebooks/02_experimental_setup.ipynb` and `src/`. The notebook was run twice on Kaggle (Tesla T4):
round 1 is in `results/runs/` (with training histories and per-image scores) and round 2 in
`results/round2/results.csv`. The slide charts are in `docs/figures/` (regenerate them with
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
- **Cost of the final model** (Tesla T4): 24.4 M parameters, 93 MB fp32 / 47 MB fp16, **about 1 hour** to train (50 epochs at 768 crops; 58–61 min in the two rounds), **0.40 s per image** end to end (raw JPEG → disk fit → network → instance map)

> *Notes:* "We select checkpoints on PQ itself because the loss does not penalise merges or splits. Scoring against every annotator and then averaging per observation means observations with three annotators don't count three times."

---

## Slide 5: Tuning, and what held up when we repeated it

Every experiment was run **twice** (two Kaggle rounds), so we could measure the noise instead of guessing it. Values are the mean of both rounds, with each round's result in brackets.

| Stage | Change | Val PQ | Verdict |
|---|---|---|---|
| A1 → A2 | fg only → all 5 outputs (U-Net from scratch) | 0.289 → **0.320** (+0.031: +0.038, +0.024) | kept, both rounds agree |
| A3 | ImageNet ResNet-34 encoder, lr 3e-4 | **0.336** (0.330, 0.342), +0.017 vs scratch | kept; also trains 1.5× faster |
| B1 | lr 1e-4 / **3e-4** / 1e-3 | 0.323 / **0.336** / 0.328 | 3e-4 best in both rounds |
| B2 | soft consensus labels (α = 0.5) | 0.335 (0.341, 0.329), −0.001 | **dropped**: the rounds disagree in sign |
| B2 | 768 crops (batch 4) | **0.355** (0.358, 0.353), +0.019 | kept |
| C | the above, 50 epochs, 3 seeds per round | **0.377 ± 0.006** (6 runs) | **final model** |
| D1–D3 | soft labels + 768; 1024 crops; 50% filament-centred crops | 0.370; 0.370; 0.362 (round 2, 20 ep) | single runs, borderline |
| D | 1024 crops, 80 epochs | 0.383 (1 seed) | within C's spread at 2.9× the training time, so stopped |
| post | fg threshold × minimum area, on C's best seed | 0.388 → **0.405** (thr 0.4, min area 200) | baseline only; the rest is in Enhancement |

Chart: `docs/figures/stage_comparison.png` shows every run as a dot, coloured by round, with C_final's spread shaded. Also `val_pq_curves.png` (Stage A plus 768 crops, round 1) and `final_seeds.png` (C_final, round 1).

What the search showed:
- **Noise level:** the same configuration scores about **±0.007** differently between runs, so two single runs need to differ by about 0.02 before it means anything.
- **Gains that held up in both rounds:** predicting the auxiliary outputs (+0.031), 768 instead of 512 crops (+0.019), the pretrained encoder (+0.017, and faster), and 50 instead of 20 epochs (+0.021).
- **A gain that didn't hold up:** soft labels gave +0.010 in round 1 but −0.013 in round 2. One run would have misled us.
- **Training-side changes plateaued at about 0.37–0.38.** Stage D's targeted changes (soft labels + 768 crops, 1024 crops, more filament-free crops) each landed within about 0.01 of C. D_final (1024 crops, 80 epochs) scored 0.383, inside C_final's spread, for 2.9× the training time. We stopped there and kept C_final.
- **The remaining errors are not a training problem:** false positives (about 870 per run, as many as the true positives) and fragmented filaments (about 130, vs about 25 merges), so RQ (0.56) is much lower than SQ (0.65). Removing blobs under 200 px alone gives +0.017, which points to post-processing (Enhancement section).
- Caveat to say aloud: the post-processing is tuned on the same validation set, so 0.405 is optimistic. The **Kaggle leaderboard** is the unbiased check.

> *Notes:* "We tuned greedily on a limited GPU budget, then ran the whole search a second time, which showed that runs of the same setup differ by about 0.007 PQ. Predicting the extra outputs, larger crops, the pretrained encoder and longer training held up in both rounds. Soft labels looked helpful once, then didn't, so we dropped them. Targeted changes after that, like larger crops and much longer training, stayed within the noise at up to three times the cost, so we stopped. The final model scores 0.377 ± 0.006 over six runs, or 0.405 after tuning the threshold. Its remaining errors are false positives and fragmented filaments, which our enhancement section tackles."

---

## Hand-offs

- **Results (section 3):** `results/runs/results.csv` (one row per run), `results/runs/<run>/val_per_set.csv` (per-image PQ, for best and worst cases), `history.csv`, and `results/runs/C_final_s1/test_pred/*.npz` (180 test instance maps from round 1's best C seed, to encode into `submission.csv`). Round 2's Kaggle output has the same from its best C seed (`C_final_s2`). Submitting both to Kaggle is a free, unbiased check of how much the choice of seed matters.
  - Best validation observations (seed 1): `20210721123430_Ch` (PQ 0.74), `20110313082654_Uh` (0.68).
  - Worst: `20220418020950_Uh` and `20200105073250_Uh` (PQ 0; one or two filaments missed), and `20140818195814_Mh` (0.10; 10 false positives).
  - By instrument, PQ ranges from 0.36 (Mh, Bh) to 0.43 (Ch).
  - Annotators disagree: on the 39 validation observations with several annotation sets, the same prediction's PQ differs by 0.14 on average between annotators. That puts a ceiling on achievable PQ and is worth mentioning.
- **Enhancement (section 4):** `src.evaluate.collect_probs` caches all five output maps for the validation set. Pass your own `postproc(probs, disk, **kw) -> instance map` to `grid_search` / `score_collected` to compare against the connected-components baseline under the same protocol. For compression, `model_size` and `benchmark` give the before/after numbers.
  - Start from the errors: false positives and fragmentation, not merges.
  - The best minimum area (200 px) was at the edge of the grid, so try larger values too.
