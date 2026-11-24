# Abstraction: triage of every method tried so far (as of 2026-11-24)

Sources: `docs/run_log.md` (base), `docs/roadmap.md` (order of 2026-11-24), main README status, method READMEs (pipeline), project memory (state, decisions ledger, philosophy), Image Atlas cards and takes. Owner marks (ledger, Atlas v6) outrank metrics on visual quality. All numbers are on the 789 held-out test tracks unless stated. r = Mantel r (Pearson) between audio-code distances and DINOv2 image-feature distances; FD = Frechet distance on DINOv2 features to held-out art (training art 113.3); "fresh reader" = top-1 naming by a reader trained afresh on frozen DINO v1 features (chance 0.13 %); "twins" = share of test images whose nearest other test image has DINOv2 cosine > 0.9 (test art 0.007; > 0.8: 0.10).

Goal used for judging (owner, 2026-11-24): the ceiling of song-to-image translation: the song recoverable from its image (fresh-reader naming), similar songs looking similar (structure), images that read as art and do not repeat, same song -> same image. Contenders must draw (decision 2026-11-22).

Status labels: **contender** (draws, on the goal), **reference/teacher** (picker, kept as yardstick or source of targets), **component** (a part that other methods use: generator, code, judge, loss term, evaluation), **parked** (did poorly, cause not shown to be fundamental; revival condition given), **dead end** (cause is fundamental; not fixable by data, generator or B-treatments).

---

## 1. Verdict table

| Method / variant | Status | Key evidence (run-log date) | Failure cause, if any | Cause type |
|---|---|---|---|---|
| **S0** (2025 cycle method) | dead end | r 0.083, below shuffled S1 control 0.140; FD 1109 (11-12). DINO v1 0.251 = MAE 0.249, i.e. at control level (11-15). Owner: blobs | cycle consistency carries no audio-distance information; CycleGAN-style is on the roadmap's excluded list | objective (fundamental) |
| **S1 + cycle term** | dead end | r 0.839 vs 0.861 without it; FD 1198 vs 1173 (11-12). Owner: blobs, "can't tell apart" | adds nothing over S1 | objective |
| **S1** (VAE, mean / standardised) | dead end as a separate method (absorbed into Q1) | r 0.861 / 0.847, kNN 0.268 / 0.410 but FD 1173 / 1364 (11-12, 11-13). Val L_struct overfits after step 5500 (11-10). Owner: blobs on every S1 grid. Pixel features see 0.14-0.28 (11-12) | latent-only distribution term does not keep outputs on the art manifold; structure term overfits 6408 tracks | objective |
| **S1 + G-GAN** (lambda 1.0, 0.3, 0.1) | dead end (same cause) | r 0.755, FD 1210, PR 11.6, collapse to red stripes (11-21); twins 0.76 (> 0.9), the worst of all systems (11-23); lambda 0.3/0.1: r 0.66, FD 1061-1091 (11-22) | swapping in a better generator did not fix it; lowering the structure weight did not fix it. Q1's feature-space realism term is the fix, which makes it Q1 | objective (generator and weight ruled out by test) |
| **Q1** (G-VAE, mean / std code) | component: the Q1 objective lives on; the VAE version is dead as a drawer | 3 seeds r 0.815-0.825, FD 409-425 (11-14); std 0.797, kNN 0.351 (11-13); shuffled 0.209, no_feat 0.869 / FD 675 (11-14); DINO v1 0.45-0.48 (mean), 0.625 (std) (11-15). Owner: mean seeds blobs, std partly | the VAE decoder: blur, "frosted glass", a third near-plain tiles | generator |
| **Q1 + G-GAN** | **contender** | r 0.537, kNN 0.108, FD 458, varied, no collapse (11-21); fresh reader 43 % vs 23 % shuffled (11-23). Owner: partly; on its shuffled control "the repetition isn't good" | repetition; weaker structure than through the VAE (0.54 vs 0.82); only 1 seed | generator range + no repetition term |
| **Q1 + CAV-MAE pull** (a 0.1 / a 0) | parked | median rank 312 vs 389, FD 538 vs 503, r 0.483 vs 0.462 (11-22); twins 0.33 / 0.31 (11-23). Owner: blobs. The a 0 run is itself weaker than Q1 + GAN (0.462 vs 0.537): batch 48, ~10k steps in 4.5 h | CAV-MAE's pair signal is weak for abstract art; run setup also under-trained | judge model (CAV-MAE) + training budget |
| **G-VAE** (both VAEs, KL sweep 0.02 to 3.0; gate) | dead end as drawing generator | prior FD 685-1261, never within 6x of art (11-10); gate passes only by degrading reconstructions (FD 208 -> 577); kl 3.0 collapses (2 active dims). Owner: every VAE grid blobs (11-19) | pixel-L1 + KL vector VAE at this capacity cannot sample art; a much stronger autoencoder would be a different generator (G-Diff/RAE) | generator family |
| **Audio code: whitened** | dead end | S1 0.272, Q1 0.259, F1 painting 0.066, PCA alignment 0.18 (11-12, 11-13); CV of distances 0.090, PR 104.7 vs art 44.8 (11-10) | near-uniform distances in more directions than art has; selected by same-track retrieval, which does not test representability in image space | code geometry (fundamental for this image space) |
| **Audio code: mean** | component | highest r in every method (S1 0.861, Q1 0.825, F1 0.641 five seeds, S2 OOS 0.710); structure sits in about 10 axes, axis 1 = acoustic vs electronic (Echo Nest acousticness rho 0.68) (11-12) | coarse: the preserved structure is genre-level | — |
| **Audio code: standardised** | component | best kNN and trustworthiness (S1 0.410, Q1 0.351) (11-13); highest under DINO v1 (Q1 0.625) (11-15); highest same-genre rate 0.528; but in S2 the code fits a Gaussian reference better than the art (z +28.4) (11-19) | lower global r than mean | — |
| Single MERT layers | dead end (minor) | no single layer exceeds the mean code after retrieval (11-13) | — | — |
| **F1** (map to DINOv2 + nearest painting) | reference / teacher | baseline painting r 0.641 (sd 0.013) mean, 0.556 std, 5 seeds; best recipes 0.747 / 0.737 (11-14, confirmed with validation selection 11-15); controls f0 0.303, shuffled 0.279, untrained 0.066; direct output r 0.945-0.988 but FD 854+ (11-13). Ceiling: sparsity of a 4666-painting gallery; 10x more paintings +0.05 (Gaussian model, 11-14). Owner: F1 grids art | a picker (owner decision 11-22); retrieval move costs structure (displacement 0.36 vs painting spacing 0.25); 385 of 789 distinct | by design (picker) |
| F1 soft / straight-through head + small chamfer | component | +0.10 r (mean), +0.085 (std) over baseline, several seed sd (11-15) | — | — |
| F1 critic, chamfer-only, ST/soft without distribution term, struct_only, views term (mean code) | dead ends (components) | critic lowers r, no FD gain in 4000 steps; chamfer and ST-without-SWD collapse to 1-63 paintings; struct_only r 0.14-0.16 (11-13, 11-14) | collapse or no gain | objective |
| F1 as teacher -> **Drawer (teach)** | parked | cos to teacher pick 0.41 (teacher's own 0.62); r 0.152 vs shuffled 0.057; fresh reader 35 % vs 29 % (11-23); + struct + penalty r 0.254, sd10 r 0.339, twins 0.12 (11-23). Owner/Atlas: very varied, no red-teal family | one picked painting per track is a weak target; the GAN can reach DINOv2 descriptions (0.78) but a second encoder sees little (DINO v1 0.41 vs 0.56 for the real painting) and composition is not carried (reach, 11-22) | generator (cannot decode DINOv2 descriptions) + teacher choice |
| **S2 transductive** (GW on test tracks) | reference | r 0.809 mean, FD 166, 789 distinct; std 0.532; restarts identical (11-19). Owner: art | solved on the test tracks; a picker | by design |
| **S2 out of sample** (nearest painting) | reference | linear r 0.718 mean, FD 448, 254 distinct; std 0.528 (11-19) | crowding onto favourite paintings | placement |
| **S2 OOS + joint assignment** | reference / teacher | r 0.710 mean, FD 225, 789 distinct; std 0.548; shuffled <= 0.13 (11-19). Owner: art ("best all-round" in takes) | picker; joint (Hungarian) step makes a track's painting depend on the other tracks in the batch, which conflicts with "same song -> same image" unless the batch is fixed | by design |
| **S2 OOS on G-GAN pool** | reference / teacher | joint r 0.672 mean, 0.583 std, FD 480-487, 789 distinct (11-19); anchored 0.685 (11-22). Owner: art | picker over generated images; FD above the generator's own samples (388) | by design |
| **S2 B-anchor** (fixed, eps 0.005) | component (works) | train->test judge-z corr 0.85-0.94 (paintings), 0.96-1.00 (GAN pool); structure cost 0.016 (alpha 0), 0.011 on the pool; settles axis 1 (agrees with D-vote, D-S4, lowest GW) and axis 2 (11-21, 11-22). First version: all 8 starts converged (11-21) | orientation does not change r or naming (metrics are reflection-invariant) | — |
| **S2 / F1 B-cluster** | parked (negative so far) | S2: K 8/16/32 r 0.486/0.530/0.567 vs flat 0.710 (mean); std 0.449-0.473 vs 0.548. F1: K 8 0.515, K 16 0.472 vs 0.646. Above random group pairing (11-21) | hard restriction to a paired group discards cross-group structure; art cloud has no strong clusters (4 styles, near-symmetric axes) | objective as built + data (few styles) |
| **F1 B-anchor** | component (works) | val/test judge-z corr 0.88; picks the axis-1 majority on both codes; anchored test r 0.624 vs seed mean 0.646 (11-21) | same as above: orientation, not quality | — |
| **D-vote** | component | axis-1 majority 92/130 = 71 % (mean), 34/50 = 68 % (std); axes 2 coin flip; axis 3 stable on mean code 119/130 (11-14) | not decisive; decided by initial weights | — |
| **D-ease / D-loss** | dead end | majority vs minority: batch distance corr 0.935 / 0.934 at step 100, val SWD 0.0190 / 0.0188, energy 0.2265 / 0.2249 (11-14) | the objective is reflection-invariant and the art features are near-symmetric on their leading axes (skew -0.06, -0.03) | fundamental |
| **D-S4** (CAV-MAE scores orientation) | component | axis 1 separated on both codes, p 0.001 / 0.002, not explained by quality (residual p 0.0004 / 0.02); axes 2-3 not separated (11-19) | — | — |
| **S4 probe** (CAV-MAE as direct bridge) | dead end as a bridge; CAV-MAE kept as judge | retrieval r 0.080 raw, 0.060 standardised (11-18); CAV-MAE vs MERT r 0.32-0.36 | CAV-MAE's sound-image similarity carries little of the MERT structure | judge model (fundamental for this model) |
| **S4 bridge** through G-GAN | dead end as a standalone bridge | median rank 136 vs 373 shuffled, top10 7 % vs 1.5 %, but r 0.063 below shuffled 0.147, FD 472 (11-22). Owner: blobs | it can be steered but steering destroys structure; same cause as the probe | judge model |
| **S3 plain** (aug) | component / ablation (superseded by S3 + structure) | top1 0.777 clean / 0.859 aug, 7.42 bits, r 0.179, FD 426 (11-22); fresh reader 52.5 % (DINO v1); window top1 0.217 vs shuffled 0.157 (11-22). Owner: partly | no structure (r 0.18) | objective (fixed by the structure term) |
| S3 noaug | dead end (variant) | 0.863 clean but 0.649 aug; fresh reader 44.7 % vs 52.5 % (11-22) | code not robust; dominated by aug | — |
| **S3 + structure term** (Kaggle, w 1.0) | **contender (lead)** | top1 0.750, r 0.513, FD 486, fresh reader 56.3 % (vs 29.3 % for S3 shuffled), window top1 0.240 (11-22, 11-23). w 0.3: r 0.222, no gain. Owner: partly, "repetition ... like collapse"; twins 0.30 | repetition; realism cost (FD 486 vs 426) | generator range + objective (fixed partly by no-repeat) |
| **S3 + structure + no-repeat** (laptop, batch 32, 12000 steps) | **contender (lead, current best balance)** | w_div 30, margin 0.8: r 0.362, FD 426, PR 29.1, twins 0.04, fresh 53.7 % vs w_div 0: r 0.399, FD 468, twins 0.12, fresh 44.7 % (11-23). w_div 100 / margin 0.7 / w_struct 0.3 all cost r and naming (11-24). Remaining twins are the most audio-backed (median audio rank 14) (11-24) | still a recurring stripe family; laptop batch halves r vs Kaggle (0.399 vs 0.513, confounded with GAN checkpoint and steps); one seed | batch/budget + generator range |
| Batch-only repetition penalty | dead end (implementation) | stayed at zero: no pair > 0.8 within 16 tracks (11-23); replaced by the bank penalty | — | — |
| **G-GAN R1, mixed precision** | component (the generator in use) | FD 492 @29k, 388 @57.6k, best 329.8 @95k, 345.7 @102k; seed 1 411 @70k (11-18, 11-19, 11-22). Owner: art (seeds 0 and 1). Random codes give twins@0.8 0.55 vs art 0.10 (11-23); reach: carries palette/texture, not composition (11-22) | limited range (repetition source for every drawer); cannot decode DINOv2 descriptions | generator capacity / data size (5832 images, 128 px) |
| G-GAN v1 (plain conv + SN), v2 SN | dead ends (configurations) | v1 FD 1094 / 1387, collapse; v2 SN FD 1472-1575, D never separated (11-15, 11-16) | configuration, superseded by R1 | — |
| Second reader, window consistency, twin share, twin-audio backing | components (evaluation) | fresh-reader baseline is the shuffled control (23-29 %), not chance (11-23) | — | — |
| Human pilot | component (evaluation, underpowered) | all systems at chance, 30/60 match; ~150-200 answers per system needed (11-19) | too few answers; images were "grey blobs" | — |
| B-time | not tried | — | — | — |

---

## 2. Treatment matrix (non-dead methods)

Probabilities are my judgement calls, not measurements. Each is the estimated chance that the treatment *materially* improves the method on the owner's goal (naming, structure, art-likeness, non-repetition), with the evidence it rests on. "n/a" = treatment does not apply.

Rows: **S3+** = S3 + structure (+ no-repeat); **Q1G** = Q1 + G-GAN; **DRW** = drawer (teach); **F1**; **S2** (OOS joint, paintings or GAN pool); **GAN** = G-GAN as a component; **PULL** = Q1 + CAV-MAE pull (parked).

### S3+ (lead contender)

| Treatment | P | Reasoning |
|---|---|---|
| Better / larger / more varied data | 45 % | S1's structure term overfit 6408 tracks (val L_struct best at 5500 steps); more art widens the GAN's range (random-code twins 0.55 vs art 0.10). Not tested; order puts it after phase 3 |
| Different generator (diffusion, flow, stronger GAN) | 55 % | Every GAN drawer repeats; the generator's own range is narrow; owner marks all drawers "partly" at best while GAN samples alone are "art". Risk: S3 trains the mapper end-to-end through the generator, so the generator must be differentiable and cheap (few-step flow/distilled diffusion, or StyleGAN2-ADA/R3GAN); 4 GB laptop cannot do it |
| B-anchor | 20 % | Works as a choice (corr 0.85-1.00 on pickers) but orientation does not move r or naming; gain is reproducibility and later human readability |
| B-cluster | 10 % | Lowered r at every K on S2 and F1 |
| B-time (sequence of windows) | 35 % | Windows give different images (window top1 24-37 %, audio 93 %); a sequence encoder would fix one image per track and give the reader more signal. Only 3 x 10 s windows exist, so little sequence to use until the data step |
| No-repeat penalty | already applied | w_div 30: twins 0.12 -> 0.04, FD 468 -> 426, fresh 45 -> 54 %, r -0.04. Further strength costs (w_div 100, margin 0.7) |
| Structure term | already applied | r 0.18 -> 0.51 at w 1.0; w 0.3 does nothing |
| Longer training / Kaggle batch 64 | 60 % | Laptop batch 32 / 12000 steps gives r 0.399 vs Kaggle batch 64 / 15000 steps 0.513 (confounded with GAN checkpoint). Kaggle runs did not hit their time limit, so the step count was a choice |
| Higher resolution | 20 % | No evidence; DINOv2 reads 112 px; may help composition and art-likeness, unlikely to help naming or structure; owner: winners only |
| Multi-seed confirmation | (confirmation) 40 % chance the laptop ranking of w_div variants flips | differences between lapd0/lapd30 (r 0.04, naming 9 points) are single-seed; Q1 seed sd ~0.005 suggests r differences of 0.04 are real, naming differences unknown. Batch 4 (seed 1) is already queued |
| As teacher / component | 40 % | Its reader is a usable naming judge; its trained mapper could provide targets for a stronger generator via distillation |

### Q1G (contender)

| Treatment | P | Reasoning |
|---|---|---|
| Data | 40 % | Same argument as S3; Q1's structure is set mostly by the code (code swap 11-12), so data helps realism more than structure |
| Different generator | 50 % | VAE -> GAN moved owner marks blobs -> partly but r fell 0.82 -> 0.54; a generator with wider range could recover structure and realism together |
| B-anchor | 20 % | Q1 seeds already agree on axis-1 sign (3/3) through the VAE; small expected gain |
| B-cluster | 10 % | negative on pickers |
| B-time | 25 % | Q1 has no naming objective, so the sequence helps mainly window consistency |
| No-repeat penalty | 65 % | Not yet applied; Q1-type systems have twins 0.31-0.33 (pull runs); the penalty transferred cleanly on S3 (twins -8 points, FD -42, naming +9) |
| Structure term | already part of Q1 | — |
| Longer training / larger batch | 35 % | The 4.5 h / batch 48 setup (a 0) gave 0.462 vs 0.537 at 20000 steps / batch 64, so budget matters; beyond 20000 steps unknown, and S1 showed structure overfitting |
| Higher resolution | 15 % | as S3 |
| Multi-seed | 15 % chance of a flip | Q1 VAE seeds were tight (r 0.815-0.825) |
| As component | 50 % | The Q1 feature-space realism term is what keeps S3 and the drawer from collapsing; it is effectively already a component of S3+ |

### DRW (drawer, parked)

| Treatment | P | Reasoning |
|---|---|---|
| Data | 20 % | Limited by what the generator can decode, not by data |
| Different generator, specifically one that decodes DINOv2 features (RAE / diffusion conditioned on DINOv2) | 40 % | Reach test: the GAN matches DINOv2 descriptions (0.78) but DINO v1 sees almost none of it (0.41 vs 0.56); a decoder built for DINOv2 would carry composition. F1's direct output keeps r 0.945-0.988 in feature space, and the interpolation curve gives r ~0.80 at FD ~425 (11-14), so a good decoder has room |
| B-anchor | 15 % | orientation is inherited from the teacher |
| B-cluster | 5 % | — |
| B-time | 15 % | teacher has no time information |
| No-repeat | already low twins (0.15); 20 % | drawer is already varied; its problem is weak linkage |
| Structure term | 45 % | r 0.152 -> 0.254 (with penalty), sd10 0.339; but then it converges towards S3+/Q1G |
| Longer training | 30 % | laptop batch 16, 6000 steps only |
| Teacher swap: S2 joint or F1 direct output instead of F1 nearest pick | 40 % | S2 gives 789 distinct targets at r 0.71; F1 direct output keeps r 0.95+; the single nearest painting is the weakest possible target (cos to own pick 0.41) |
| Higher res / multi-seed | 5 % / n/a | — |

### F1 (reference / teacher)

| Treatment | P | Reasoning |
|---|---|---|
| Larger gallery | 70 % small gain | Gaussian model: 10x paintings +0.05, 100x +0.08 (11-14) |
| Different generator | n/a as picker; as teacher see DRW | — |
| B-anchor | done; works (corr 0.88), no quality gain | — |
| B-cluster | done; negative | — |
| B-time | 20 % | mapping generalises well (train 0.965 / test 0.945); time could add information |
| Others | n/a | F1 is a reference; its five-to-fifty seeds already exist |
| As teacher | 35 % | conditional on a DINOv2-decodable generator; through the current GAN only +6 naming points |

### S2 (reference / teacher)

| Treatment | P | Reasoning |
|---|---|---|
| Larger / more varied data | 50 % | GW fits the art shape better than a Gaussian on the mean code (z -9.2) but not on the standardised code; more styles may add shape for GW to lock onto; 3000-track subset used so far |
| Different generator (pool from a better generator) | 50 % for art-likeness | pool FD 487 vs paintings 225; a better generator pool closes that gap, but still a picker |
| B-anchor | done; works (cost <= 0.016) | — |
| B-cluster | done; negative | — |
| B-time | 15 % | — |
| As teacher for a drawer | 40 % | best per-track targets available (distinct, r 0.71, art-like); untested; caveat that joint assignment is batch-dependent, so targets must be computed once on the training set |

### GAN (component)

| Treatment | P | Reasoning |
|---|---|---|
| Longer training | 50 % (small-medium) | FD 388 -> 346 (best 330) from 57.6k to 102k steps, noise ~30 between evaluations; diminishing |
| Stronger GAN (StyleGAN2-ADA, R3GAN) or diffusion | 60 % | 5832 images is small-data territory these were built for; random-code twins 0.55 vs art 0.10 |
| More art data | 55 % | standard for GANs; owner order allows moving data up if a new generator needs it |
| Higher resolution | 25 % | 128 px now; doubles cost; owner: winners only |

### PULL (Q1 + CAV-MAE pull, parked)

All treatments <= 15 %, except "use CAV-MAE only as B-anchor judge" (already works). The pull moved median rank 389 -> 312 at FD +35; the cause is the judge model, which no listed treatment changes.

### Ranked (method, treatment) pairs

Score = P x gain size (gain: 1 small, 2 medium, 3 large; large = blobs/partly -> art, or > 0.15 r, or > 15 naming points). Judgement calls.

| Rank | Pair | P | Gain | Score | Note |
|---|---|---|---|---|---|
| 1 | S3+ x new generator (fast differentiable: StyleGAN2-ADA/R3GAN or few-step flow) | 55 % | 3 | 1.65 | biggest lever on repetition and art-likeness; owner step 1 |
| 2 | Q1G x new generator | 50 % | 3 | 1.50 | same lever; also tests whether structure lost VAE -> GAN comes back |
| 3 | Q1G x no-repeat penalty | 65 % | 2 | 1.30 | cheap, laptop-sized, direct transfer from S3 evidence |
| 4 | S3+ x Kaggle batch 64 / full steps (with no-repeat) | 60 % | 2 | 1.20 | separates the batch effect from the penalty effect |
| 5 | Drawer x DINOv2-decodable generator (RAE / DINOv2-conditioned diffusion), taught by S2 joint or F1 direct output | 40 % | 3 | 1.20 | only route by which the strong pickers become drawers |
| 6 | GAN x stronger architecture or more art data | 60 % | 2 | 1.20 | feeds every drawer; overlaps with 1-2 |
| 7 | S3+ x larger / more varied data | 45 % | 2.5 | 1.13 | owner step 4 |
| 8 | GAN x longer training (checkpoint 102k, FD 346) | 50 % | 1.5 | 0.75 | already available; switch changes every result, so do it in the reruns |
| 9 | S3+ x B-time | 35 % | 2 | 0.70 | owner step 3 |
| 10 | Q1G x longer training / batch | 35 % | 2 | 0.70 | — |
| 11 | S2 x as teacher (without new generator) | 25 % | 2 | 0.50 | reach test caps it through the current GAN |
| 12 | S3+ / Q1G x higher resolution | 15-20 % | 2 | 0.3-0.4 | winners only |
| 13 | S3+ / Q1G x B-anchor | 20 % | 1 | 0.20 | needed for identifiability and human judging, not metrics |
| 14 | any x B-cluster | 5-10 % | 1-2 | <= 0.2 | negative on both pickers |

---

## 3. Dead ends (cause shown to be fundamental)

1. **S0 / cycle consistency (and the S1 cycle term).** r 0.083 below a shuffled control; the term adds nothing to S1 (0.839 vs 0.861). The objective carries no audio-distance information, and CycleGAN-style translation is excluded by the roadmap. No data or generator change supplies the missing signal.
2. **S1 as a separate method (latent-only distribution term).** Collapses with the VAE (FD 1173-1364) and with the GAN (FD 1210, twins 0.76); the structure weight is ruled out (lambda 0.1/0.3: FD 1061-1091). The generator change has been tested and failed. The only fix is the feature-space realism term, which is Q1. It lives on inside Q1 and S3, not as its own row.
3. **Whitened audio code.** Fails in all four methods (0.07-0.27). Near-uniform distances (CV 0.090) over 105 effective directions cannot be represented in an art space with 45. This is a property of the code relative to the image encoder, not of data volume.
4. **G-VAE as the drawing generator** (both VAEs, whole KL sweep, the gate rule). Every grid is marked blobs by the owner; prior samples never come within 6x of art FD; the gate can be passed only by degrading reconstructions. A stronger autoencoder would be a different generator (G-Diff / RAE), so this counts as dead for G-VAE as built.
5. **D-ease and D-loss.** The two orientations learn at identical speed and reach identical losses. The objective is reflection-invariant and the art cloud is near-symmetric, so no amount of data or training will separate them with these tests. D-vote, D-S4 and B-anchor already do the job.
6. **S4 as a standalone bridge (probe retrieval and the GAN bridge).** CAV-MAE can be steered (median rank 136) but r falls to 0.06, below shuffled (0.15); direct retrieval r 0.06-0.08. The limit is what CAV-MAE's similarity encodes about music (r 0.32-0.36 with MERT). Neither data, generator nor B-treatments change the judge model. CAV-MAE stays as a judge (D-S4, B-anchor).
7. **Dead configurations and variants** (not methods): G-GAN v1 and the spectral-norm GAN; S3 without augmentation (dominated: 65 % under augmentation, fresh reader 45 % vs 52 %); batch-only repetition penalty (never fired); F1 critic / chamfer-only / straight-through without a distribution term / struct_only (collapse or loss); single MERT layers.

### Parked (poor so far, not shown to be fundamental)

| Method | What would revive it | Owner's "better data" category? |
|---|---|---|
| Drawer (teach) | a generator that can decode DINOv2 descriptions (RAE / DINOv2-conditioned diffusion); a teacher with distinct targets (S2 joint) or F1's direct output; a target read by two encoders | partly: generator first, then data |
| B-cluster | a soft group prior (coarse-to-fine start, or a group term) instead of hard restriction; a more varied art set with real clusters (now 4 styles, near-symmetric cloud) | **yes**: the clearest case where more varied art could change the result |
| Q1 + CAV-MAE pull | a stronger text-free audio-visual model, or a fair rerun at the full Q1 budget (the a 0 baseline was itself below Q1 + GAN) | no |
| S3 + lighter structure (w 0.3), w_div 100, margin 0.7 | larger batch or a wider generator may move the trade-off; retry when generator changes | weakly |
| F1 critic | ran only 4000 steps; revive only if a longer schedule is cheap | no |
| Q1 through the VAE | only as a structure ceiling reference (r 0.82 at FD 410-425); as a drawer it needs the generator replaced | no |
| G-GAN seed 1 | behind seed 0 (411 at 70k); a second generator seed is needed for the multi-seed stage | — |

---

## 4. Caveats

- **Single seeds.** Everything except Q1 through the VAE (3 seeds), F1 retrieval (5 to 50 seeds) and the GAN (2 seeds) is one seed. S3+, Q1G, the drawer and every no-repeat variant are single-seed. The owner decided (2026-11-19, 2026-11-24) that multi-seed runs come at the end for winners. Rankings among close laptop variants (r differences of 0.04, naming differences of 5-9 points) are not confirmed. Batch 4 (seed 1 of lapd30 and lapd0, shuffled control with the penalty) is the first check.
- **Laptop vs Kaggle confounds.** Laptop S3 runs use batch 32 and 12000 steps on the improved GAN (step 101996, FD 346). The Kaggle S3 run uses batch 64 and 15000 steps on the older GAN (step 57568, FD 388). The r gap (0.399 vs 0.513) cannot be assigned to batch alone. The Q1 bridge runs at batch 48 under a 4.5 h limit were weaker than the batch-64 Q1 run.
- **Mantel r depends on DINOv2**, which is also in the training losses. Under pixel features the structure is 0.02-0.31. DINO v1 sees 55-80 % of the DINOv2 value, and MAE features do not discriminate at all (PR 1.6). r on the mean code and r on the standardised code are not comparable (each method keeps the structure of the code it was trained on; code-to-code r 0.27 for mean vs whitened).
- **FD is on DINOv2 features and does not track the owner's eye.** Q1 through the VAE has FD 409-425 and is marked "blobs". Q1 through the GAN has FD 458 and is marked "partly". The S3 shuffled control has FD 390 and is marked "art". Q1 no_feat raises r while the images become haze. Owner marks outrank FD for art-likeness.
- **The fresh-reader baseline is the shuffled control (23-29 %), not chance**: a smooth mapper makes tracks distinguishable without audio information. The window-consistency and twin-audio measures have the same smoothness floor (shuffled window top1 16 %, drawer-control twin rank 52).
- **The twin share uses a fixed DINOv2 threshold (0.8 / 0.9).** Every GAN system sits far above real art at 0.8 (about 0.55-0.68 vs 0.10), partly because of the generator's own range.
- **Same song -> same image.** Mappers and generators are deterministic. S2 joint assignment and B-cluster group restriction depend on which other tracks are in the batch, so as pickers or teachers their targets must be fixed once, on the training set.
- **Human evaluation is underpowered.** The pilot was at chance for all systems with 20 trials per system; about 150-200 answers per system are needed to detect 0.58. No drawing system has been judged by humans yet.
- **Data scale.** 7994 tracks of 30 s (3 windows), 5832 paintings in 4 styles at 128 px, plus photos of objects and rooms that stay in the set by owner decision. B-time has only 3 windows to work with until the data step.
- **Probabilities in section 2 are judgement calls**, each tied to the evidence cited next to it. They are meant for ordering the work, not as forecasts.
