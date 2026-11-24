# Abstraction: combination analysis (chains of treatments), 2026-11-24

Builds on `docs/triage.md` (single-treatment verdicts and probabilities) and `docs/generator_options.md` (generator families G1-G4, D1-D7, O1-O6). Evidence is cited by run-log date (`docs/run_log.md`, base). All probabilities are judgement calls tied to the evidence next to them; they are for ordering work, not forecasts. "Proposed" marks ideas of mine that are not the owner's decisions. Abbreviations as in triage: r = Mantel r on DINOv2 features (mean code unless stated), FD on DINOv2 to held-out art, fresh = fresh-reader top-1 on DINO v1, twins = share of test images whose nearest other test image has DINOv2 cosine > 0.9 (twin@0.8 for > 0.8).

Cost legend: **L** = laptop, hours to a day; **K** = inside one Kaggle week (<= 30 GPU h); **K+** = more than one Kaggle week; **X** = beyond the current budget.

---

## 0. Failure causes the analysis is built on

Every interaction below is argued from one of these measured causes.

| Id | Cause | Evidence (run-log date) | Systems it binds |
|---|---|---|---|
| C1 | **Generator range floor.** Random G-GAN codes already give twin@0.8 0.55 against real art 0.10 | 11-23 (batch 1 and twin table) | every drawer |
| C2 | **Generator reach.** The GAN can match a DINOv2 description (0.78) but a second encoder sees little of it (DINO v1 0.41 vs 0.56 for the real nearest painting); palette and texture carried, composition not | 11-22 (reach) | drawer, any painting teacher, structure ceiling of every drawer |
| C3 | **Structure-realism trade-off curve** in DINOv2 space: direct F1 output r 0.945 at FD 854 down to 0.645 at FD 384; r about 0.80 at FD about 425 | 11-14 (F1 limits A); Q1 no_feat r 0.869 / FD 675 (11-14); S3 struct w 1.0 FD 486 vs 426 (11-22) | all |
| C3' | **GAN drawers sit far below that curve**: Q1+GAN 0.537 at FD 458, S3+struct 0.513 at FD 486, lapd30 0.362 at FD 426, while the curve gives roughly 0.80-0.87 at FD 425-485 and Q1 through the VAE sat on it (0.825 at 425) | 11-14, 11-21, 11-22, 11-23 | the headroom a better decoder could recover (upper bound under DINOv2; the owner marked the on-curve VAE output "blobs") |
| C4 | **Smooth mapper merges moderately similar tracks.** Twins come from audio neighbours (median audio rank 12-36 vs 394) but 55-70 % are not in each other's top 10; shuffled controls reach fresh 23-29 % and window top-1 16 % | 11-24 (twin backing), 11-23, 11-22 (windows) | all mappers |
| C5 | **No-repeat vs structure conflict at the generator floor.** w_div 30 brings twin@0.8 to 0.55 (= the random-code floor) at r -0.04; going below the floor (w_div 100: 0.48; margin 0.7: 0.41) costs r (0.31-0.32) and naming (46 %); remaining twins are the most audio-backed (rank 14) | 11-23 (batch 2), 11-24 (batch 3, twin backing) | S3+, any drawer with the penalty |
| C6 | **Batch / budget.** Laptop batch 32 / 12000 steps r 0.399 vs Kaggle batch 64 / 15000 steps 0.513 (confounded with GAN checkpoint); Q1 bridge at batch 48 / 4.5 h r 0.462 vs 0.537 | 11-23, 11-22 | S3+, Q1G |
| C7 | **Structure overfits with the decoder in the loop**: S1 val L_struct best at step 5500; F1 (no decoder in loop) generalises, train 0.965 / test 0.945 | 11-10 (S1 lambda), 11-13 (F1) | through-generator mappers; not yet checked for S3 |
| C8 | **Art cloud near-symmetric and weakly clustered** (skew -0.06 / -0.03 on leading axes, 4 styles); D-ease/D-loss identical curves; the most skewed axis (mean-code axis 3) is the one consistently oriented, 119/130 | 11-14 | B-cluster, D-ease, D-loss, orientation |
| C9 | **Hard group restriction discards cross-group structure**: B-cluster r 0.486-0.567 vs flat 0.710 (S2), 0.472-0.515 vs 0.646 (F1); pairing still beats random pairing, gap growing with K | 11-21 | B-cluster |
| C10 | **Gallery sparsity** for pickers: 10x paintings +0.05 r (Gaussian model) | 11-14 | F1, S2 as teachers |
| C11 | **Teacher target quality.** Single F1 pick: drawer cos 0.41, +6 naming points over its shuffled control | 11-23 | drawer |
| C12 | **Judge model.** CAV-MAE vs MERT r 0.32-0.36; S4 bridge r 0.063 < shuffled 0.147 | 11-18, 11-22 | S4, pull |
| C13 | **Information is not the binding limit for naming.** S3's own reader names 75-78 % of 789 test tracks; fresh readers 53-56 %; audio windows are 93 % nearest neighbours but images only 22-25 % | 11-22 | B-time, music data |
| C14 | **Batch-dependent assignment** (S2 joint, B-cluster groups) conflicts with same song -> same image unless fixed once on the training set | triage caveats, 11-19 | S2 teacher, B-cluster |

One structural fact matters for every chain: **S3 + structure contains Q1G.** S3+struct = Q1's structure term on DINOv2 features + latent SWD + DINOv2-feature SWD + InfoNCE reader (methods/s3/README, run 11-22). Q1G uses the same three Q1 terms with somewhat different weights (w_lat 0.25, global + local structure) and no reader. The measured difference is what the reader buys: fresh 56 % vs 43 %, r 0.513 vs 0.537, FD 486 vs 458 (11-22, 11-23). Q1G is therefore close to the w_info = 0 arm of S3+, and most treatments act on both through the same cause.

---

## (a) Interaction matrix

Codes: **C** complementary (different causes, gains should add), **S** substitute (same cause, gains overlap), **P** prerequisite (row or column only pays after the other; arrow shows direction: `P>` = row needed first, `P<` = column needed first), **X** conflict, **0** independent / no evidence of interaction.

Treatments: GAN+ = stronger GAN (R3GAN loss, W mapping net; G1/G2); DEC = conditional decoder on own representation (Diff-AE / RCG style, D5/D6); FLOW = one-step flow from scratch (O3/O1); ART = more / more varied art; MUS = more music (FMA-medium/large); ANC = B-anchor; CLU = B-cluster hard; CLUs = B-cluster soft (group prior); TIME = B-time; NR = no-repeat penalty; ST = structure term; B64 = Kaggle batch 64 / longer; INFO = S3 reader (InfoNCE); T-POOL = S2 joint assignment over a GAN pool as drawer targets; T-PNT = S2 joint over paintings (or F1 pick) as targets.

| | DEC | FLOW | ART | MUS | ANC | CLU | CLUs | TIME | NR | ST | B64 | INFO | T-POOL | T-PNT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GAN+** | S | S | C | 0 | 0 | 0 | 0 | P> | P> | C | C | C | C | S |
| **DEC** | | S | P< | C | 0 | 0 | C | C | C | C | 0 | C/X | 0 | P> / X |
| **FLOW** | | | P< | 0 | 0 | 0 | 0 | P> | P> | C | C | C | C | S |
| **ART** | | | | 0 | C | P> | P> | 0 | C | 0 | 0 | 0 | C | C |
| **MUS** | | | | | 0 | 0 | 0 | P> | 0 | C | C | C | C | C |
| **ANC** | | | | | | S | S | 0 | 0 | 0 | 0 | 0 | P> | P> |
| **CLU** | | | | | | | S | 0 | X | X | 0 | 0 | X | X |
| **CLUs** | | | | | | | | 0 | C | X (weak) | 0 | 0 | 0 | 0 |
| **TIME** | | | | | | | | | 0 | C (weak) | 0 | C | 0 | 0 |
| **NR** | | | | | | | | | | X | C | C (to w 30) / X (beyond) | C | C |
| **ST** | | | | | | | | | | | C | C | S | S |
| **B64** | | | | | | | | | | | | C | 0 | 0 |
| **INFO** | | | | | | | | | | | | | C | C |
| **T-POOL** | | | | | | | | | | | | | | S |

### Reasons for the non-trivial cells

| Pair | Relation | Reason (cause, evidence) |
|---|---|---|
| GAN+ / DEC / FLOW | S | All three replace the generator; all attack C1. DEC alone also attacks C2 (it is trained to reproduce paintings from a whole-image representation). Pick one family per round. |
| GAN+ - ART | C | Two different sources of the C1 floor: the loss (mode dropping, what R3GAN targets) and coverage of the training set. Real test art has twin@0.8 0.10 (11-23), so the floor of 0.55 is not the data's own repetition; both levers can lower it from different sides. |
| DEC, FLOW - ART | P< | Memorisation evidence (generator_options §0): ~4.7k training images at 128 px sits below the generalisation regime for diffusion/flow models. ART must come first. |
| GAN+, FLOW - TIME | P> | C13: the reader already names 75-78 %; extra audio information cannot show up in the image until the generator can carry more. TIME before a wider generator mostly feeds a channel that is already saturated. |
| GAN+, FLOW - NR | P> | C5: the penalty has already pushed twin@0.8 to the generator's random-code floor (0.55 = 0.55). Every stronger setting tried (w_div 100, margin 0.7) went below the floor and cost r and naming. A lower floor is needed before a stronger penalty can pay. |
| GAN+ - ST | C | C3': drawers sit 0.3+ r below the feature-space curve at equal FD. That gap is attributed to the decoder (C2), not to the structure term; a wider generator should let the same ST weight buy more r. |
| GAN+, any generator - B64 | C | Different causes (C6 vs C1/C2). Confound warning: every new-generator run must be compared at the same batch and steps, or the C6 confound of 11-23 repeats. |
| DEC - MUS, DEC - ST | C | With DEC the mapper is trained in representation space without the generator in the loop (generator_options §2.4). F1 shows such mappers generalise (train 0.965 / test 0.945, 11-13), while through-decoder structure overfits (S1, 11-10). More music then helps directly, and ST in c-space can approach the F1 direct-output values (r 0.945 in feature space, 11-14). |
| DEC - INFO | C / X | C: the reader term can be computed on c. X: a reader on c can name tracks from c even if the decoded images do not differ (cycle-fidelity gap, generator_options §2.4). Requires the fresh reader on pixels -> DINO v1 as the arbiter. |
| DEC - T-PNT | P> / X | P>: painting teachers only pay with a decoder that can draw a description (C2; through the GAN, +6 points, 11-23). X: a decoder that memorises, taught with exact painting representations as targets, reproduces the picked painting: the drawer becomes a picker in disguise, which the owner ruled out (decision 2026-11-22). Needs nearest-training-image distance as a gate and targets that are not exactly training-painting codes. |
| DEC - CLUs | C | A per-group prior in c-space is cheap (no generator in the loop) and DEC trained on varied art is the setting where clusters can exist. |
| ART - CLU, CLUs | P> | C8/C9: 4 near-symmetric styles give k-means partitions of a near-continuous cloud. Group pairing carries information (beats random pairing at every K, 11-21) but less than the flat match. B-cluster can only gain where the art has real clusters. |
| ART - ANC | C | More varied art is likely less symmetric (C8); orientation then has consequences for realism, and the anchor choice matters more. |
| MUS - TIME | P> | FMA-small, -medium and -large are all 30 s clips [M: from memory of the FMA paper]; only fma_full has whole tracks (hundreds of GB [M], beyond laptop disk). A real sequence needs re-embedding with finer windows (e.g. 2.5-5 s) or fma_full. With 3 x 10 s windows, B-time has little sequence to use (triage). |
| MUS - ST | C | C7: the structure term overfit 6408 tracks with the decoder in the loop. Whether S3+ also overfits is unchecked (free diagnostic, section e). |
| ANC - CLU, CLUs | S | Group pairing fixes the coarse assignment (which audio region goes to which art region), which is the job the anchor already does (S2 B-anchor cost 0.011-0.016 r, 11-21, 11-22). |
| ANC - T-POOL, T-PNT | P> | A drawer inherits its teacher's orientation. Anchoring the teacher (S2 anchored on paintings and on the GAN pool already exists, 11-21, 11-22) gives an anchored drawer for free. |
| CLU - ST | X | C9, measured on two pickers. |
| CLU - NR | X | Restricting tracks to a group's subset of image space concentrates them, which raises look-alikes; the penalty then fights the restriction. Not measured; follows from C9 and C5. |
| CLU - T-POOL / T-PNT | X | Hard-restricted teachers give lower-r targets (0.486-0.567 vs 0.710). |
| CLUs - ST | X (weak) | A soft group term pulls toward paired clusters; if the pairing is wrong (random pairing r 0.29-0.43) it costs structure, as hard B-cluster did, but by less at small weight. |
| CLUs - NR | C | Proposed form: replace the single global feature SWD with a per-group SWD (each audio group's images must spread like its paired art cluster). That forbids collapse inside a region, which the global SWD permits, so it attacks C1/C4 from a different side than the penalty. |
| TIME - INFO | C | Only the reader objective rewards extra information; ST and NR do not. |
| TIME - training augmentation | X | Training now samples one window per track, which doubles as augmentation and gives window robustness (window top-1 0.40-0.46 for S3, 11-22). A sequence encoder over all windows removes that variation (6408 sequences vs 19224 windows) and invites C7-type overfitting. Needs window dropout / random sub-sequences. |
| NR - ST | X | C5 and twin backing (11-24): the twins left after w_div 30 are the most audio-backed; removing them removes structure. |
| NR - INFO | C to w_div 30, X beyond | w_div 30: fresh 45 -> 54 %; w_div 100 or margin 0.7: 46 % (11-23, 11-24). |
| NR - B64 | C | Different causes. Hyper-parameters do not transfer automatically: the bank (1024 images) refreshes twice as fast at batch 64 and w_div 30 was tuned at batch 32, so the Kaggle run needs a small w_div bracket. |
| ST - INFO | C | S3 w 1.0 keeps naming (75 % vs 78 %) and gains r 0.18 -> 0.51; fresh 53 -> 56 % (11-22). The cost is realism (FD +60). |
| ST - T-POOL, T-PNT | S | Teacher targets carry the same audio geometry as the structure term. Evidence: drawer + struct + w_div 10 (sd10) r 0.339 vs S3+struct + w_div 10 (lapd10) r 0.351 (11-23); the teacher adds nothing on top of ST through the current GAN. |
| INFO - T-POOL | C | The pool teacher gives reachable, distinct, structure-carrying targets; the reader adds distinguishability. Not measured. |
| T-POOL - GAN+ | C | Pool images are the generator's own outputs, so they are reachable by construction (bypasses C2). A better generator gives a better pool. |
| T-POOL - T-PNT | S | Both are S2 joint assignments; the pool version trades art FD (480-487 vs 225, 11-19) for reachability. |
| CAV-MAE pull - anything | 0 / X | C12 is untouched by every treatment in the list; the pull cost FD +35 for median rank 389 -> 312 (11-22). |
| RES (higher resolution) - everything | P< | Owner: winners only. Also P< generator: the GAN at 128 px already has the C1 floor, and higher resolution needs more data for any GAN or diffusion model. DINOv2 reads 112 px, so the DINOv2 metrics will not show most of a resolution gain. |

---

## (b) Bottleneck order per contender, teacher and parked method

"Binds first" = the limit that, while present, hides the effect of the later treatments.

| System | 1st bottleneck | 2nd | 3rd | 4th | What this means for ordering |
|---|---|---|---|---|---|
| **S3 + structure + no-repeat** (lapd30: r 0.362, FD 426, twins 0.04, twin@0.8 0.55, fresh 54 %; Kaggle w 1.0 without penalty r 0.513) | **C6 batch/budget**: the laptop config has never run at batch 64; without it every later comparison carries the 11-23 confound | **C1 range floor**: penalty already at the floor (C5); recurring stripe family | **C2/C3' reach**: r 0.36-0.51 vs ~0.8 curve | C7 overfitting (unchecked for S3), then C13 makes B-time and more music secondary | Batch-64 rerun of lapd30 on the improved GAN is the baseline every generator test needs; then generator; data and B-time only after the generator |
| **Q1 + G-GAN** (r 0.537, FD 458, fresh 43 % vs 23 %; twins not measured) | **C1/repetition** (owner: "the repetition isn't good"); NR never applied | **C2/C3'**: r fell 0.82 (VAE) -> 0.54 (GAN) | C6 (bridge setup at batch 48 gave 0.462) | C7 | Q1G is nearly the w_info = 0 arm of S3+: test NR and new generators on it as one extra arm of each S3+ run, not as its own line |
| **Drawer (teach)** (cos to pick 0.41, r 0.152, fresh 35 % vs 29 %) | **C11 teacher target** (one unreachable painting per track) | **C2 reach** for any painting teacher | ST substitution (teacher adds nothing on top of ST, 11-23) | C1 | Swap to a reachable teacher first (T-POOL, laptop). Painting teachers wait for DEC |
| **F1 as teacher** | C2 reach (through the GAN) | C10 gallery sparsity | — | — | Only useful as teacher after DEC; F1 direct output (off-manifold, FD 854) as a target makes the drawer a Q1 variant (S with ST) |
| **S2 as teacher** (OOS joint r 0.710 on paintings, 0.672 on the GAN pool, 789 distinct) | C14 (assignment must be fixed on training tracks; capacity: 6408 training tracks vs 4666 paintings or 5372 kept pool images) | C2 (painting version only) | C10 | — | Pool version is usable now: generate a pool >= 6408 kept images and fix one joint assignment on the training set |
| **G-GAN (component)** | C1 range (loss / mode coverage) | data size (5832 images, 128 px) | C2 composition | — | R3GAN loss first (works on current data); varied-art pretraining as the complementary lever |
| **B-cluster** | C9 hard restriction | C8 no real clusters in 4 styles | — | — | Hard form: no chain revives it on drawers. Soft form needs varied art |
| **B-anchor** | none for pickers (works); for drawers: no mechanism yet beyond seed selection | — | — | — | Proposed inference-time version (section c, chain 3) removes the cost |
| **B-time** | C13 (image cannot carry more) | data: 3 windows only | window-sampling augmentation conflict | — | After generator; full version after music re-embedding |
| **Q1 + CAV-MAE pull** | C12 judge | C6 under-trained setup | — | — | No available treatment touches C12; keep CAV-MAE as judge only |

---

## (c) Chain table

P = chance the chain materially improves on the current best for its target (stated per chain); gain 1 small, 2 medium, 3 large (as in triage: large = owner marks move from partly toward art, or > 0.15 r, or > 15 naming points at equal twins).

| # | Chain | Expected effect | P | Gain | Cost | Depends on | Cheapest first test of the key assumption |
|---|---|---|---|---|---|---|---|
| 1 | **Drawer taught by S2 joint assignment over a GAN pool** (T-POOL), fixed once on the training tracks, + L_lat + L_feat + NR; variants + ST, + INFO (proposed) | Reachable, distinct, structure-carrying targets: attacks C11 and bypasses C2. Teacher scores r 0.672 / 0.685 anchored, 789 distinct, owner "art" for the pool picks (11-19, 11-22). Drawer will land below the teacher; the question is whether it lands above lapd30 on r at similar twins | 35 % (beats lapd30 on r at twins <= 0.05) | 2.5 | L | Pool >= 6408 kept images from the improved GAN (gan_pool.py, minutes); S2 joint on the training tracks (oos.py); targets.npz in the teach.py format (DINOv2 + DINO v1 of the assigned pool image, plus its z) | Without the generator in the loop: regress the mapper onto the targets' DINOv2 features and z (F1-style, minutes), decode test tracks, measure r, twins and DINO v1 cosine to the assigned image. Risk to watch: z regression averages targets and shrinks z (bland, repetitive images), which L_lat must hold off |
| 2 | **S3+ lapd30 config at batch 64 / 15000 steps on the improved GAN (step 101996), with a Q1G + NR arm (w_info 0) and a small w_div bracket** | Separates C6 from the penalty effect; gives the baseline for all generator tests; first Q1G-with-penalty number | 60 % (r recovers toward the 0.51 of Kaggle w 1.0 with twins <= 0.05) | 2 | K (about 3 h per arm; S3 runs of 15000 steps stayed under 2.75 h, 11-22) | Kaggle reset (by 11-27) | Seed-1 laptop pair (batch 4, running) tells whether the lapd30 vs lapd0 ranking holds before spending Kaggle hours |
| 3 | **B-anchor at inference by reflecting the code's top-3 axes** (8 sign patterns fed to an already-trained drawer; CAV-MAE judge z on validation tracks; pick the best) (proposed) | Reflection is an isometry of the code, so r, kNN and naming capacity are unchanged; gives every drawer an anchored orientation with no retraining. Fulfils the owner's step 2 for B-anchor on all contenders at near-zero cost | 70 % (judge choice stable val -> test, corr >= 0.8 as on pickers: 0.85-1.00, 11-21, 11-22) | 1 | L (CAV-MAE image embeddings via `s4/extract.py --images`) | Codes must be roughly symmetric on those axes, or reflected codes are out of distribution for the mapper | Skew of the top-3 code axes on training tracks (seconds); if skewed, compare FD / twins of reflected vs original outputs before judging. Fallback: anchor by seed selection at the multi-seed stage |
| 4 | **R3GAN loss (+ W mapping net) on the G-GAN -> S3+ (NR, batch 64) and the Q1G arm** | Lower C1 floor; with it the penalty can be raised without the C5 cost; some of the C3' gap | 45 % | 2.5 | K+ (the current GAN took about 20 GPU h over three jobs to reach step 101996, 11-18/19/22; R2 adds cost; bridges then need a second week) | Chain 2 as baseline at equal batch and steps | Laptop debug at small width, then on Kaggle compare random-code twin@0.8 and FD with the current GAN at an equal step count. Before that, real-art twin@0.8 per style (minutes): if one style is itself twin-heavy, a share of the floor is data |
| 5 | **Chain 4 + varied-art pretraining** (other WikiArt styles for pretraining, fine-tune on the owner's abstract set) | Complementary attack on C1 (coverage vs loss) | 55 % for the GAN range, conditional on 4 | 2.5 | K+ (data step partly moved up: owner's rule allows it when a generator needs it) | Data acquisition. Note [M]: the four classes appear to match the counts of the common 27-style WikiArt dump (Abstract Expressionism 2782, Colour Field 1615, Minimalism 1337, Action painting 98), so "more abstract art" may need other sources; "more varied" means figurative styles, which risk pulling outputs away from abstraction unless used only for pretraining | Check the dump's class counts against the set (no download beyond metadata); then FD / twins of a fine-tuned vs from-scratch GAN at equal steps |
| 6 | **Conditional decoder on own representation (Diff-AE style D6; RCG style D5 only with an encoder that is not DINO v1) + varied art + mapper trained in c-space with the S3+ losses (or taught by S2 on paintings) + novelty gate, optional distillation to one step** | The only chain that attacks C2 directly; makes mappers F1-like (no generator in the loop, so C7 should vanish and MUS pays); ceiling is the F1 curve (C3') | 30 % | 3 | K+ to X | ART first (memorisation); cycle fidelity E(G(c)) close to c on mapper outputs; DINO v1 must stay out of training (it is the fresh-reader encoder) | Small 64 px Diff-AE on the current 4666 paintings (one Kaggle session): decode interpolations between pairs of training c and measure nearest-training-image distance and DINO v1 similarity. If interpolations snap to training paintings, the decoder is a picker and the chain needs more data first |
| 7 | **One-step shortcut flow (O3) as drop-in for G-GAN -> S3+** | Mode coverage without an adversarial loss; every bridge unchanged | 25 % | 2.5 | K+ | ART (memorisation, blur risk at 5k) | Random-sample twin@0.8 and nearest-training distance of a small model on current data |
| 8 | **B-time (3-window sequence encoder with window dropout) on S3+** | Mainly window consistency (22-25 % now); naming gain capped by C13 | 25 % now; 35 % after chain 4 or 6 | 1.5 | L to K | Generator (C13); real sequences need re-embedding (MUS) | Train S3+ on laptop with the sequence code, compare fresh reader and window top-1 against lapd30 at the same batch |
| 9 | **More music (FMA-medium) for S3+ / Q1G** | Attacks C7 if S3 overfits; more naming negatives | 35 % | 2 | K (MERT embedding on Kaggle, retrain) | Disk: embeddings are 3 x 13 x 768 per track, about 120 KB at float32, so 25k tracks about 3 GB [E]; fits Kaggle, tight on the laptop | Free: check whether S3's validation structure loss and validation InfoNCE turn upward during training in existing logs. No upturn = data is not the first limit |
| 10 | **Soft B-cluster: per-group feature SWD (group prior) on S3+** (proposed) | Forbids within-region collapse; complements NR | 10 % now; 20 % with varied art | 1.5 | L | ART (C8) | Cluster quality of art features (silhouette or gap statistic, current vs varied art), minutes, before any training |
| 11 | **Projected GAN (G3) as generator** | Strong small-data art results in the paper | 35 % for range | 2 | K | **Conflict**: with DINO v1 features it contaminates the fresh reader; MAE mean-patch features are near one-dimensional (PR 1.6, 11-15); EfficientNet is ImageNet-label supervised (owner's call) | Owner decision on the feature network before any run |
| 12 | **Retry S3 light structure (w 0.3), w_div 100, margin 0.7 on the new generator** | These failed through C5 at the current floor; a lower floor may move the trade-off | 30 % | 1 | cheap arms inside chain 4 | Chain 4 | — |
| 13 | **Drawer taught by F1 direct output** | Becomes a Q1 variant (teacher and ST substitute) | 15 % | 1 | L | — | Not worth a run before chain 1 |
| 14 | **Higher resolution on the winner** | Composition and art-likeness for the eye, little on DINOv2 metrics | 20 % | 1.5 | K+ | Winner, generator, data | — (owner: winners only) |
| 15 | **Q1 + CAV-MAE pull at full Q1 budget** | Would only show whether the setup or the judge limited it | 10 % | 1 | K | — | None needed; C12 dominates |
| 16 | **Hard B-cluster on drawers** | Expected to lower r as on both pickers | 5 % | 1 | L | — | Skip; one run only if the owner wants the row filled for "B-cluster on all contenders" |

### Answer to the owner's example: S3 + structure x bigger data x B-cluster

- S3+ and bigger data are complementary through two causes: C7 (if S3 overfits; unchecked) and C1 (more varied art lowers the generator floor). Most of the value of bigger data reaches S3+ through the generator, so it pays most when the generator is retrained on it (chains 5, 6).
- B-cluster in the hard form conflicts with the structure term (C9) and with no-repeat; it would subtract. In the soft form it is a different realism term, plausible only on art with real clusters.
- Expected: S3+ x bigger data (via the generator) about 45 %, medium-to-large gain; adding soft B-cluster on top adds about 10-20 % chance of a small gain; adding hard B-cluster is expected to cost r. The triple chain's value is almost all in its data part.

---

## (d) Dead ends revisited

| Dead end | Failure cause | Does any available solution touch the cause? | Verdict |
|---|---|---|---|
| **S0 / cycle consistency** (r 0.083 < shuffled 0.140, 11-12; S1 + cycle 0.839 vs 0.861) | objective carries no audio-distance information; CycleGAN-style is on the excluded list | No. The useful part of the cycle idea (recover the input from the image) already lives in S3's reader, which is an information-based cycle and works (78 % naming, 11-22). Data, generator and B-treatments add no distance signal to the cycle loss | **Still dead** (its sound part is absorbed into S3) |
| **S1 standalone** (latent-only distribution term; collapse with VAE and GAN; lambda ruled out, 11-21, 11-22) | matching z to N(0, I) does not keep images varied, because image change per unit of z is uneven across the GAN's input space | In principle a generator with path-length regularisation (StyleGAN2 W space, G2) makes latent spread track image spread more closely, which is the mechanism of the collapse. But a revived S1 can at best reach Q1G (S1 = Q1G minus the feature realism term), so it has no value as a method | **Still dead as a method.** Optional free arm on a new W-space generator as a probe of latent uniformity (about 15 % that it no longer collapses) |
| **Whitened code** (0.07-0.27 everywhere; CV of distances 0.090; PR 104.7 vs art 44.8, 11-10, 11-12) | near-uniform distances in more directions than the art space has | Varied art could raise the art space's effective dimension, which shrinks the dimensional mismatch. It does not change the near-uniform distances: with CV 0.090 there is little distance variation to preserve, and the standardised code already covers fine structure | **Still dead** (5-10 % via varied art; not worth a run; an evaluation-only PR check on varied art is free when the data step happens) |
| **G-VAE as drawing generator** (prior FD 685-1261; owner: blobs, 11-10, 11-19) | pixel-L1 + KL vector VAE at this capacity cannot sample art | A stronger autoencoder (L1 + adversarial, spatial latent) is the first stage of latent diffusion (D3), but that is a different generator. Q1 through the VAE keeps one use: the on-curve structure reference (0.825 at FD 425) | **Still dead** as drawer; lives on as a reference and, rebuilt, as the AE stage of D3 |
| **D-ease / D-loss** (identical curves and losses, 11-14) | the structure term is reflection-invariant and the art cloud is near-symmetric (skew -0.06, -0.03) | **Yes, partly.** The feature-distribution terms are not reflection-invariant when the art is skewed. Evidence for the mechanism: the most skewed axis (mean-code axis 3) is the one that orients consistently (119/130, 11-14). More varied art is likely to be less symmetric, so D-loss (realism term at convergence) could separate orientations | **Revivable via more varied art, about 30 %.** Low value: it would choose by shape, which S2's lowest-GW choice and B-anchor already provide, and they agree on axis 1 (11-21). Free first test: skew of leading DINOv2 axes on the varied art set |
| **S4 standalone bridge** (r 0.063 < shuffled 0.147; probe r 0.06-0.08, 11-18, 11-22) | what CAV-MAE's similarity encodes about music (r 0.32-0.36 with MERT) | No. A new generator lets CAV-MAE steer harder, but the structure it steers toward is still CAV-MAE's. No other text-free audio-visual model with weights is available (DenseAV excluded for its transcript-tuned HuBERT; MAViL weights unconfirmed). Training an own audio-visual model on video is beyond the budget | **Still dead.** CAV-MAE stays as judge (D-S4, B-anchor, chain 3) |
| G-GAN v1, spectral-norm GAN | configuration | superseded by R1 and by R3GAN (R1 + R2) | still dead |
| S3 noaug | code hides in patterns augmentation breaks (86 -> 65 %, 11-22) | no treatment changes that trade | still dead |
| Batch-only repetition penalty | never fires: no pair > 0.8 within 16 tracks (11-23) | batch 64 gives more pairs per batch, but the bank version already does the job | still dead |
| F1 critic, chamfer-only, straight-through without SWD, struct_only | collapse or no gain | F1 is a reference now; no value in reviving | still dead |
| Single MERT layers | no layer beats the mean code (11-13) | E-other-music later is the relevant test, not layers | still dead |

### Parked items and their revival chain

| Parked | Revival chain | When |
|---|---|---|
| Drawer (teach) | chain 1 (pool teacher) now; chain 6 with S2 painting teacher after art data | now / after data |
| B-cluster | soft per-group realism term x varied art (chain 10) | step 2 (one laptop run, low prior), real test at step 4 |
| Q1 + CAV-MAE pull | none available (C12); keep CAV-MAE as judge | — |
| S3 w 0.3, w_div 100, margin 0.7 | retry as cheap arms on the new generator (chain 12) | with chain 4 |
| F1 critic | none worth running | — |
| Q1 through the VAE | structure ceiling reference only | — |
| G-GAN seed 1 | multi-seed stage | end |

---

## (e) Ranked list and proposed order

### Ranked by P x gain / cost weight (L = 1, K = 2, K+ = 3, K+ to X = 5)

| Rank | Chain | P x gain / cost | Note |
|---|---|---|---|
| 0 | Evaluation-only diagnostics: Q1G twin share; real-art twin@0.8 per style; skew of top code axes and of art axes; art cluster quality; S3 validation curves; WikiArt class counts | — (minutes, no training) | each settles an assumption of chains 1-10 before money is spent |
| 1 | 1. Drawer taught by S2 over a GAN pool | 0.35 x 2.5 / 1 = 0.88 | laptop, code mostly exists |
| 2 | 3. B-anchor by code reflection | 0.70 x 1 / 1 = 0.70 | laptop; fulfils owner step 2 for anchors |
| 3 | 2. S3+ lapd30 at batch 64 + Q1G-NR arm | 0.60 x 2 / 2 = 0.60 | baseline for generator step |
| 4 | 4. R3GAN (+W) -> S3+ / Q1G arm | 0.45 x 2.5 / 3 = 0.38 | owner step 1 |
| 5 | 11. Projected GAN | 0.35 x 2 / 2 = 0.35 | blocked on the feature-network decision |
| 6 | 5. R3GAN + varied-art pretraining | 0.55 x 2.5 / 4 = 0.34 | moves part of data step up |
| 7 | 8. B-time on S3+ | 0.25 x 1.5 / 1.5 = 0.25 | 0.35 after a new generator |
| 8 | 9. More music | 0.35 x 2 / 3 = 0.23 | depends on diagnostic |
| 9 | 12. Retry parked S3 variants on new generator | 0.30 x 1 / 1.5 = 0.20 | free arms |
| 10 | 6. Conditional decoder + varied art | 0.30 x 3 / 5 = 0.18 | highest ceiling; largest unknowns |
| 11 | 7. One-step flow | 0.25 x 2.5 / 4 = 0.16 | after data |
| 12 | 10. Soft B-cluster | 0.10-0.20 x 1.5 / 1 = 0.15-0.30 | depends on art clusters |
| 13 | 13. F1 direct-output teacher | 0.15 | substitute for Q1G |
| 14 | 14. Higher resolution | 0.10 | winners only |
| 15 | 15, 16 | <= 0.05 | skip |

### Proposed order (consistent with the owner's order of 2026-11-24; deviations flagged)

**Now, laptop, until Kaggle resets (by 2026-11-27)**
1. Let batch 4 finish (seed 1 of lapd30 / lapd0, shuffled + penalty).
2. Diagnostics of rank 0 (evaluation only).
3. Chain 1 first test (regression without the generator), then the through-generator drawer with L_feat + NR if the first test holds. *Flag: this is a teacher swap on a parked method, not in the owner's list; it uses idle laptop time and does not displace step 1.*
4. Chain 3 (reflection anchor) on lapd30, Q1G and the drawer. *Flag: this is owner step 2 done before step 1. It is inference-only, so it cannot be invalidated by step 1 and costs nothing to repeat on the new generator.*
5. Write and debug the R3GAN loss (+ mapping net) at small width on the laptop.

**Kaggle week A (owner step 1: generators)**
- GPU 0: R3GAN generator on the current art (about one week of the GPU-0 share at the current GAN's pace).
- GPU 1: chain 2 (lapd30 at batch 64 on the improved GAN; Q1G + NR arm; w_div bracket). *Flag: run Q1G as an arm of S3+ rather than a separate line, since S3+ contains Q1G's terms; saves Kaggle hours.*

**Kaggle week B**
- S3+, Q1G arm, chain-1 drawer (if it passed on the laptop) and the chain-12 arms on R3GAN, at the same batch and steps as week A.
- Decision point: if R3GAN lowers random-code twin@0.8 and S3+ follows, continue with the GAN family; if not, the next generator is a diffusion or flow family, and by the owner's rule **the art-data step moves up** (chains 5-7). *Flag: the evidence (C1, the memorisation threshold, and the observation that the current four classes may already be the whole of the common WikiArt dump for those styles) suggests preparing the varied-art set during week A anyway, because both the GAN-pretraining lever and every non-GAN generator need it.*

**Owner step 2 (B-anchor and B-cluster on all contenders)**
- B-anchor: done by chain 3 on the final generator.
- B-cluster: one soft-form run on the best contender; hard form skipped on drawers. *Flag: hard B-cluster lowered r at every K on both pickers (11-21) and conflicts with ST and NR; its only plausible version needs varied art, so its real test belongs at step 4.*

**Owner step 3 (B-time on the best)**
- 3-window sequence encoder with window dropout. *Flag: with C13 and only three windows, expect a small effect; the full B-time test needs finer-window re-embedding, which fits naturally into step 4.*

**Owner step 4 (larger, more varied data)**
- Music: FMA-medium (re-embed with finer windows to serve B-time as well). Art: varied set if not already done.
- Retry on the new data: soft B-cluster, D-loss orientation check (revivable dead end), S3 light-structure / margin variants, drawer with an S2 painting teacher through a conditional decoder (chain 6) if it was not already started.

**Owner step 5, then winners**: S5, S6, D-human, D-complexity; higher resolution on the winner; multi-seed for winners (with seed selection by the CAV-MAE judge as the fallback anchor).

---

## Caveats

- **Single seeds.** S3+, Q1G, the drawer and every no-repeat variant are one seed. The differences on which C5 rests (r 0.04-0.09, naming 8-9 points) are not yet confirmed; batch 4 is the first check.
- **Laptop vs Kaggle confound.** Laptop S3 runs use batch 32 / 12000 steps on the GAN at step 101996; the Kaggle S3 run used batch 64 / 15000 steps on step 57568. Chain 2 exists to remove this before any generator comparison.
- **DINOv2-dependent metrics.** r and FD are on DINOv2, which is also in the training losses. The C3' headroom (curve vs drawers) is an upper bound under DINOv2: the on-curve Q1 VAE output was marked "blobs". Any generator or teacher that uses DINO v1 in training (Projected GAN with DINO v1, RCG with DINO v1) removes the only independent reader; MAE mean-patch features are not a usable replacement (PR 1.6).
- **Owner marks outrank FD.** S2 pool picks are "art" at FD 480-487 while some drawers at FD 426-458 are "partly": chain 1's success criterion should include the Atlas, not only r and twins.
- **Same song -> same image.** Chain 1 targets, B-cluster groups and the reflection sign must be fixed once on training data; diffusion/flow decoders need a fixed x_T and an ODE sampler; B-time needs a fixed window scheme. The no-repeat penalty acts only in training.
- **Items marked [M]** (FMA subset lengths, WikiArt class counts) are from memory and should be checked before planning the data step; [E] estimates are mine.
- **Probabilities are judgement calls**, each tied to a cause in section 0. They order work; they are not forecasts.
