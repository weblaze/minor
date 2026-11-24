# Generator options for Abstraction (G-GAN successors, G-Diff and others)

Prepared 2026-11-24. Scope: image generators that a music-driven mapper can control, trained or fine-tuned with no text anywhere, and evaluated on the project's constraints (differentiable path, determinism, ~5k images at 128 px, GTX 1650 / Kaggle 2 x T4).

Conventions in this report:
- **[V]** = checked this session on arxiv.org (abstract page, HTML, or the arXiv API) or on the official repository page.
- **[M]** = from memory of the paper or code, not re-checked this session. Treat as likely but unverified.
- **[E]** = my own estimate (compute, expected behaviour). No paper reports it.
- "Text contamination" covers three levels: **none** (self-supervised or own training), **label** (ImageNet class labels: human labels, but not text and not a text encoder), **text** (CLIP, captions, text-to-image weights; excluded by the project's rule). A weaker form, **filter**, means the training images were selected with a text-trained model (for example, CLIP aesthetic scores), but the weights never saw text.

---

## 0. Background that drives the recommendations

1. **Diffusion memorises on small data.** Kadkhodaie et al. (arXiv 2310.02557 [V]): CelebA 80 x 80 UNet denoisers memorise at N = 1 to 100 images, sit in a transition regime at N = 1,000 (samples look like training images with distortions), and generalise at about N = 10^5. Larger images need more data, though not proportionally. Akbar et al. (2305.07644 [V]) found diffusion memorises more than StyleGAN on small medical datasets. Shah et al. (2502.21278 [V]) report that memorisation grows as the training set shrinks, and that training only at high noise levels (ambient-diffusion style) reduces it without hurting quality. **With 4,666 training images at 128 px, a diffusion model trained from scratch will probably sit between the transition and the generalisation regimes [E].** Its failure mode is the reverse of the GAN's: varied outputs that are near-copies of training paintings, not near-copies of each other. The project must measure the nearest-training-image distance alongside the twin rate.
2. **The gradient cost depends only on the number of network evaluations between the mapper's output and the pixels.** A one-step generator costs one forward and backward pass, as the GAN does now. An N-step ODE sampler costs N, unless the mapper is trained in a feature space and the generator is never differentiated (Section 2).
3. **Determinism.** Every candidate below is deterministic given fixed noise. Diffusion and flow models need a deterministic ODE sampler (DDIM 2010.02502 [V], Euler or Heun on the probability-flow ODE, EDM 2206.00364 [V]) and a fixed seed for x_T. Stochastic samplers (ancestral DDPM, SDE) are not deterministic unless their per-step noise is also fixed.
4. **Evaluation leakage.** If a discriminator, perceptual loss or condition uses DINOv2, the FD-on-DINOv2 metric is partly being optimised directly. Any option that adds a pretrained feature network should be scored with a second encoder (DINO v1, MAE, or the planned E-2nd-image) as well.
5. **Compute yardstick [E].** At fp16 a T4 runs at roughly half a V100. The EDM repo reports about 2 days on 8 x V100 for CIFAR-10 at 32 px and about 4 days on 8 x V100 for FFHQ or AFHQv2 at 64 px ([V], github.com/NVlabs/edm), so 16 to 32 V100-days. One Kaggle week (~30 T4-hours) is about 0.6 V100-days. A full EDM-scale pixel model at 128 px is therefore far beyond one week; small models, pretrained starts, or latent spaces are needed. StyleGAN2-ADA at 128 px on one V100: 4 h per 1,000 kimg, 7.2 GB, "5,000 kimg often gives reasonable results" ([V], NVlabs/stylegan2-ada-pytorch README), so about 20 V100-hours, roughly 40 T4-hours or 1.3 Kaggle weeks [E].
6. **Hardware note.** The float16 NaN problem is a GTX 1650 issue. T4s support fp16 tensor cores. They have no native bf16 [M]. Mixed precision on Kaggle is worth validating; the current G-GAN runs fp32 at 2.1 s per step (gan README).

---

## 1. Candidate table

Abbreviations: MD = what the mapper drives; Diff = differentiable path for mapper training; Det = determinism; Data = behaviour with ~5k images at 128 px; Fit = compute fit (L = laptop, K = one Kaggle week, K+ = several weeks, X = beyond); Lic/weights = licence and public non-text weights; Text = contamination level.

| # | Family | Method | Key paper | MD | Diff (cost) | Det | Data (~5k, 128 px) | Fit | Lic / weights | Text |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | GAN | Current G-GAN + R3GAN loss (relativistic pairing + R1 + R2) | R3GAN, 2501.05441 [V] | 128-d z | yes, 1 pass | yes | good; built to fix mode dropping; keep DiffAugment (2006.10738 [V]) | L for debugging, K | R3GAN/ code MIT; rest NVIDIA Source Code License; weights FFHQ 64/256, CIFAR, ImageNet 32/64 under CC BY-NC(-SA) ([V], github.com/brownvc/R3GAN) | none (if trained from scratch) |
| G2 | GAN | StyleGAN2-ADA (mapping net, W / W+ space) | 2006.06676 [V] | w (512-d) or W+ | yes, 1 pass | yes | designed for "a few thousand" images; ADA | K+ (about 1.3 weeks to 5,000 kimg [E]) | NVIDIA Source Code License (non-commercial) [V]; weights FFHQ, MetFaces, AFHQ, BreCaHAD, CIFAR [V] | none (FFHQ/AFHQ starts are photos, no text) |
| G3 | GAN | Projected GAN / FastGAN generator | 2111.01007 [V], 2101.04775 [V] | z | yes, 1 pass | yes | strong: Art Painting (1,000 imgs) FID 27.96 vs 43.07 for StyleGAN2-ADA; Landscape (4,319) 6.92 vs 15.99 [V] | L/K (paper: "5 days to under 3 hours" for equal FID) | MIT ([V], autonomousvision/projected-gan) | **label**: default feature net EfficientNet-Lite1, ImageNet-supervised [V]. The paper also tested R50-CLIP (**text**, must not be used) [V]. Swapping in DINO v1 / MAE is untested by the paper. |
| G4 | GAN | StyleGAN-XL | 2202.00273 [V] | z + class | yes | yes | poor fit: built for large class-conditional data; relies on ImageNet classifier guidance | X | MIT [V] | **label**: EfficientNet-Lite0 + DeiT-base discriminator features, DeiT-S classifier guidance, class embeddings from EfficientNet [V]; no CLIP in the paper [V] |
| G5 | GAN | Vision-aided GAN (ensemble of pretrained discriminators) | 2112.09130 [V] | z | yes | yes | made for limited data [M] | K | [M] | depends on the chosen backbones: CLIP is **text**; DINO/Swin options [M] |
| D1 | Diffusion, pixel | EDM/ADM-style UNet, unconditional or conditioned, DDIM/Heun ODE sampler | EDM 2206.00364 [V]; DDIM 2010.02502 [V] | x_T noise (3x128x128) and/or condition c | yes but N steps (N = 18 to 50 Heun/DDIM; 2N evals for Heun) | yes with ODE | memorisation risk (Section 0); Patch Diffusion (2304.12526 [V]) reports gains down to ~5,000 images and >= 2x faster training | K+ to X from scratch at 128 [E]; K for a small model at 64 px [E] | EDM: CC BY-NC-SA 4.0, weights CIFAR-10 32, FFHQ 64, AFHQv2 64 [V] | none |
| D2 | Diffusion, pixel, x-prediction | JiT (plain ViT on pixels, large patches, x-prediction) | 2511.13720 [V] | x_T, c | N steps | yes with ODE | untested at this scale; x-prediction is the change that makes pixel ViTs work [V] | K+ [E] | code status not verified | none |
| D3 | Latent diffusion, own AE | Self-trained autoencoder (KL or VQ, L1 + LPIPS + GAN) + DiT/UNet in latent | LDM 2112.10752 [V]; DiT 2212.09748 [V] | x_T latent (e.g. 16x16x4), c | N steps through the latent model + 1 decoder pass | yes | the AE trains well on 5k [E]; the latent prior has the same memorisation risk | K (AE + small DiT at 16x16 latent) [E] | own weights | LPIPS uses VGG (ImageNet **label**) [M]; or use L1 + own GAN only |
| D3b | Latent diffusion, public AE | Stable Diffusion VAE (sd-vae-ft-mse) as frozen autoencoder | model card [V] | as D3 | as D3 | yes | AE quality on art probably good at 128 px (16x16 latent) [E] | K | MIT [V] | AE itself is image-only, but the original kl-f8 was trained on OpenImages and ft-MSE was fine-tuned on LAION-Aesthetics + LAION-Humans [V]; LAION-Aesthetics is selected with CLIP-based scores [M] -> **filter** level. The owner must decide whether that is allowed. |
| D4 | Diffusion in a representation space | RAE: frozen self-supervised encoder + trained ViT decoder; DiT with wide DDT head in that space | 2510.11690 [V] | latent tokens, c | N steps + decoder | yes | not shown on small data; DiT width must match token dim [V] | K+ [E] | code status [M] | encoders tested: DINOv2 (none), MAE (none), **SigLIP2 (text, exclude)** [V]. MAE-B reconstructs best (rFID 0.16 vs 0.49 DINOv2-B) but DINOv2 generates best [V]. |
| D5 | Representation-conditioned generation | RCG: generate a self-supervised representation, then a pixel generator conditioned on it | 2312.03701 [V] | the representation vector r (MoCo v3 256-d [M]) | **no backprop needed** if the mapper is trained in r-space; optional through-generator fine-tune | yes (ODE + fixed noise) | the pixel generator becomes a conditional decoder; 5k is thin [E] | K for a small conditional model at 64 to 128 px [E]; official recipes use 4 to 128 V100s [V] | MIT; ImageNet weights [V] | MoCo v3 is self-supervised (none) [V]; replace with DINO v1, MAE or own encoder |
| D6 | Diffusion autoencoder | Diff-AE: learned semantic encoder z_sem + diffusion decoder; stochastic part x_T from DDIM inversion | 2111.15640 [V] | z_sem (512-d [M]) and/or x_T | as D5 | yes (DDIM) [M] | encoder and decoder trained jointly on the art; reconstructs, so memorisation is expected but z_sem interpolations are smooth [M] | K at 64 px; K+ at 128 [E] | MIT [M] | none (own encoder; gets the project out of DINOv2 space) |
| D7 | Self-guided diffusion | CFG on self-supervised features / clusters instead of labels | 2210.06462 [V] | condition c from self-supervised features | as D1 (2x evals per step with CFG) | yes | [M] | K+ | [M] | none if a DINO / MoCo-type encoder is used |
| F1 | Flow matching / rectified flow | Same networks as D1-D6 trained with flow-matching loss; straighter paths, fewer steps; reflow for near-one-step | 2210.02747 [V], 2209.03003 [V] | x_0 noise, c | N Euler steps (often 10 to 50 [M]); reflow 1 to 2 steps | yes (ODE) | as D1 | as D1 | own | none |
| O1 | One-step, from scratch | MeanFlow / improved MeanFlow | 2505.13447 [V], 2512.02012 [V] | noise (and c) | **yes, 1 pass** | yes | untested at 5k; training needs a JVP (torch.func.jvp in the official PyTorch port [V]) | K+ [E] (paper scale: ImageNet, TPU; py-meanflow trains CIFAR-10 on 8 x H200 [V]) | JAX official + PyTorch CIFAR-10 port (Gsunshine/py-meanflow [V]; licence not read) | none for the loss |
| O2 | One-step, pixel | pixel MeanFlow (pMF) | 2601.22158 [V] | noise, c | yes, 1 pass | yes | untested at 5k | X at paper scale (batch 1024, 160 to 360 epochs, TPU) [V]; small B/16 variant at 128 px is K+ [E] | code release not stated [V] | **label**: perceptual losses LPIPS-VGG and a ConvNeXt-V2 variant (weights 0.4 / 0.1) [V]; class-conditional with CFG in the paper [V] |
| O3 | One-step / few-step | Shortcut models (step-size-conditioned flow) | 2410.12557 [V] | noise, c | yes, 1 to 4 passes (choose at inference) | yes | untested at 5k; no JVP, single training phase [V] | K [E] at 64 to 128 px with DiT-S/B | MIT, JAX, CelebA-HQ 256 + ImageNet 256 checkpoints [V] | none |
| O4 | Consistency models | CT / iCT / ECT / sCM | 2303.01469 [V], 2406.14548 [V], 2410.11081 [V] | noise, c | yes, 1 to 2 passes | yes | ECT starts from a pretrained diffusion model, so it inherits D1's data behaviour [M] | ECT: K from a diffusion start [E]; sCM: X | openai/consistency_models MIT; ImageNet-64 (class-cond.), LSUN Bedroom / Cat 256 checkpoints [V] | CD-LPIPS variants use LPIPS (VGG, **label**) [V]; CT / L2 variants none |
| O5 | Distillation | Distribution Matching Distillation (one-step student of a diffusion teacher) | 2311.18828 [V] | noise, c | yes, 1 pass | yes | needs a good teacher first | teacher + distil: K+ [E] | — | none if the teacher is own |
| O6 | One-step, feature-space training | Drifting models | 2602.04770 [V] | noise, c | yes, 1 pass | yes | loss computed in a pretrained self-supervised feature space (MoCo v2, SimCLR, own latent-MAE) [V]; effective batch 4,096 to 8,192 with in-batch positives and negatives [V] | X at paper batch sizes; unknown at small batch [E] | code not seen | none with MoCo / SimCLR / own MAE |
| T1 | Normalising flow | TarFlow | 2412.06329 [V] | latent z (same size as image) | yes, but autoregressive over patches: many sequential passes | yes | unknown | K+ / X [E] | [M] Apple | none |
| A1 | Token models | VQGAN + transformer; MaskGIT | 2012.09841 [V], 2202.04200 [V] | condition vector (prefix / class slot) | **no** through discrete sampling (argmax/categorical); Gumbel or straight-through only | yes with greedy / fixed-seed decoding | the transformer overfits on 5k token sequences [E] | K | MaskGIT weights ImageNet [M] | none (VQGAN uses LPIPS-VGG, label) |
| A2 | Continuous AR | MAR (AR + per-token diffusion head) | 2406.11838 [V] | condition vector | many steps (AR steps x diffusion-head steps) | yes with fixed noise | unknown | K+ | MIT [M] | none |

Pretrained, publicly available, non-text starting points (all **label** or **none**; none is art-trained):
- EDM FFHQ-64 / AFHQv2-64 unconditional, CIFAR-10 (CC BY-NC-SA) [V].
- OpenAI consistency models / EDM on ImageNet-64 (class-conditional) and LSUN Bedroom / Cat 256 (MIT code) [V].
- StyleGAN2-ADA FFHQ / MetFaces / AFHQ [V]; R3GAN FFHQ-64/256, ImageNet-32/64 [V].
- RCG ImageNet MoCo v3 + RDM + MAGE/DiT/ADM/LDM [V].
- Shortcut models CelebA-HQ 256 and ImageNet 256 [V].
- Stable Diffusion VAE (sd-vae-ft-mse, MIT): the AE only; **filter**-level caveat [V]/[M].
- Excluded: any Stable Diffusion / SDXL / FLUX / PixArt denoiser (text-conditioned); StyleGAN-T (CLIP) [M]; Vision-aided GAN with CLIP backbones; Projected GAN with R50-CLIP; RAE with SigLIP2.

---

## 2. How to train a mapper through each family

The current recipe has four loss groups: an InfoNCE reader on image features, a structure term, distribution terms, and a no-repeat penalty. They all need d(features of image)/d(mapper output). There are five ways to get that gradient.

### 2.1 One-step generators (G1-G4, O1-O6): unchanged recipe
GANs, MeanFlow / pMF, shortcut models at 1 step, consistency models and distilled students are all a single network call z -> x. The mapper is trained exactly as through G-GAN today: cost is one generator forward and backward per batch. The only difference is the latent's shape:
- GAN: 128-d z; StyleGAN: w in W (512-d), which is better conditioned for a mapper than z [M].
- One-step flow / consistency: the "latent" is image-shaped noise (3x128x128 or the AE latent). Let the mapper output a low-dimensional code u and map it through a fixed linear lift (random orthogonal matrix or low-frequency basis), then renormalise to the Gaussian shell (norm about sqrt(D)). A distribution term should keep the lifted noise Gaussian-typical; off-shell noise gives washed-out or saturated images [M]. Adding a condition input c gives the mapper a second, more semantic control (see 2.4).

### 2.2 Multi-step ODE samplers (D1-D7, F1): backprop through N steps
Options, cheapest first:
1. **Few steps + full backprop.** With a flow-matching or EDM model, 8 to 16 Euler/Heun steps already give usable images [M]. Memory grows linearly with N. At 128 px with a small UNet (about 30 to 60 M parameters), a mapper batch of 16 to 32 on a 16 GB T4 is plausible at N = 8 [E].
2. **Gradient checkpointing per step** (torch.utils.checkpoint around each denoiser call). Memory drops to one step's activations plus N stored x_t; compute is about 2x forward. This is the standard route used by DRaFT-style work [M].
3. **Truncated backprop (DRaFT-K).** Run the first N-K steps without gradient and backprop only through the last K (DRaFT 2309.17400 [V], which also shows K = 1 works with a variance-reduction variant). For this project it is a poor fit: if the mapper drives x_T or a condition used mainly in early steps, truncation cuts off exactly the steps that set composition. Fine for colour and texture terms only.
4. **Adjoint methods** (AdjointDPM 2307.10711 [V]; Adjoint Matching 2409.08861 [V]). Constant memory and exact gradients for the ODE, at about 2x to 3x the compute of a forward solve [M]. Worth it only if N must stay large.
5. **D-Flow** (2402.14017 [V]) optimises the source noise per sample by differentiating through the flow and shows this keeps the result on the data manifold. The mapper is the amortised version: it learns a function from songs to noise in place of per-song optimisation. The paper supports mapping into the initial noise of a deterministic flow sampler.

Cost estimate [E]: training the mapper through N = 8 steps with checkpointing costs about 16 to 24 generator evaluations per batch, against 2 for the GAN. With a generator about the size of G-GAN, one mapper run stays inside a Kaggle session.

### 2.3 Distil first, then train the mapper (D* -> O4/O5)
Train or fine-tune the diffusion model, distil it to one or two steps (ECT 2406.14548, DMD 2311.18828, or train a shortcut model directly), then use 2.1. This is the cleanest way to have diffusion-quality coverage and a GAN-like mapper cost. Risk: one-step students lose diversity or detail compared with the teacher [M]. Measure the twin rate on the student, not the teacher.

### 2.4 Train the mapper on a feature-space target (D5, D6, D7, any conditional model)
If the generator is conditioned on a vector c that is itself an image representation, train the mapper entirely in c-space and never differentiate the generator:
- c = E(image) for a self-supervised encoder E (DINO v1, MAE, MoCo v3) or, better for the owner's goal, the Diff-AE semantic encoder trained with the generator (D6). With the Diff-AE encoder the representation space belongs to the generator, not to DINOv2.
- The reader, structure and no-repeat terms are computed on c (or on a small head over c). The distribution terms keep c near the cloud of real E(paintings). A representation generator, the RCG RDM, can serve as a learned prior, with its denoising score as a regulariser [M].
- At inference: image = G(c, fixed x_T) with a deterministic ODE. The cost of mapper training is that of the existing F1 bridge, which already maps into image-feature space. This is a generative version of F1.
- A required check: cycle fidelity E(G(c)) ≈ c on mapper outputs, not only on real-image embeddings. Without it the reader may "name" songs in c-space while the images do not differ. A short through-generator fine-tune (2.2 or 2.3) can close that gap afterwards.
- Classifier-free guidance on c (train with c dropped 10 to 20 % of the time) gives a single knob between faithfulness to the song's c and image quality [M]. Self-guided diffusion (2210.06462 [V]) shows that self-supervised features can replace labels for guidance. CFG doubles the evaluations per step.

### 2.5 Inversion as a supervision source
Deterministic ODE samplers are invertible: DDIM inversion gives each real painting an x_T, and a GAN can be inverted by optimisation. This gives (a) a real-art prior in the mapper's control space for the distribution terms, and (b) a way to check whether the mapper's outputs sit in the region that real paintings occupy. For one-step flows, inversion needs a few-step ODE run of the underlying flow [M].

### 2.6 Token models (A1, A2)
Gradients do not pass through discrete token sampling. The mapper can only drive the conditioning vector, and must be trained on a feature-space target (2.4) or with straight-through / Gumbel tricks, which are noisy. MAR avoids discrete tokens but stacks AR steps and diffusion-head steps, the most expensive path in this list. Low priority for this project.

### 2.7 Adapters
Adapters (ControlNet / IP-Adapter / LoRA) are attractive mainly on large pretrained text-to-image models, which are excluded. The non-text equivalent: take a class-conditional ImageNet model (consistency-model ImageNet-64, DiT, EDM), replace the class-embedding lookup by a learned projection of c, and fine-tune on the art. The mapper then drives the same adaLN slot that used to hold the class embedding. The ImageNet prior carries **label** contamination and natural-image bias; with 4.7k paintings it may still pull outputs toward objects [E]. Licences are mostly non-commercial.

---

## 3. Ranked recommendation

### Rank 1: R3GAN loss on the existing G-GAN, then a StyleGAN2-style mapping network (G1, then G2-lite)
- **Why:** cheapest change with the most direct attack on the measured failure (55 to 65 % twins). R3GAN's stated aim is to fix mode dropping and non-convergence with a regularised relativistic loss and no tricks [V]. The interface (one pass, 128-d or 512-d latent) and every existing bridge stay as they are. A mapping network to W gives the mapper a better-conditioned space [M]. Runs on Kaggle in about one week; can be debugged on the laptop.
- **Expected gain [E]:** lower twin rate, maybe toward 25 to 40 %, from random latents; similar art-likeness (FD in the same range or better); little gain in composition, because the 128 px resolution and the 5k images limit layout learning in any GAN.
- **Fails if:** the twins come from the data (many near-duplicate Colour Field / Minimalism paintings) rather than the loss. Check the real-art twin rate per style first. It also fails if relativistic pairing destabilises with DiffAugment at batch 64 (not reported in the paper; R3GAN's README configurations use batch 256 or more [V]).
- **Data:** works on current data.

### Rank 2: a conditional flow-matching / diffusion decoder on the project's own representation, mapper trained in that space (D6 Diff-AE style, or D5 RCG style with DINO v1 / MAE), then distilled or a shortcut variant for through-generator fine-tuning
- **Why:** this is the option that most changes what the image can carry. The condition c encodes the layout and semantics of a whole painting; the generator is trained to reproduce paintings from c, so mapper-chosen c values can produce composition, not only palette and texture. The mapper trains as cheaply as F1 (2.4). With a Diff-AE-style encoder trained jointly, the space is the project's own, which leaves DINOv2 behind. Deterministic through a fixed x_T and an ODE sampler.
- **Expected gain [E]:** largest gain in composition and range: distinct c gives distinct images by construction, and twins should fall toward the real-art 10 % if c is spread. Art-likeness is uncertain, since FD at 128 px from a small model trained from scratch may be worse than the GAN's 346 at first.
- **Fails if:** (a) the decoder memorises: with near-unique c per painting and 4.7k images, G(c) can become a lookup of training paintings, so mapper outputs between training c values decode to blends or to the nearest painting. Measure nearest-training distance; (b) cycle fidelity E(G(c)) ≈ c is poor for off-manifold c; (c) compute: at 128 px it needs a small UNet or a 16x16 latent with an own AE (D3).
- **Data: needs more and better data.** Memorisation evidence (Section 0) puts 4.7k images below the generalisation regime. Pretrain on the full WikiArt collection (all styles, image only, no labels used) or other non-text image sets, then fine-tune on abstract art. **Move the bigger-data step earlier for this option.** Ambient-style high-noise training (2502.21278) and Patch Diffusion (2304.12526) are mitigations worth including.

### Rank 3: a one-step flow trained from scratch: shortcut model first, MeanFlow / iMF if it works (O3, O1), in own-AE latent (D3) or pixels at 64 to 128 px
- **Why:** a drop-in replacement for G-GAN as a noise-to-image map in one pass, so every existing mapper and bridge works unchanged (2.1), with the mode coverage of a likelihood-style objective, not an adversarial one. Shortcut models need no JVP and no distillation stage, and their code and checkpoints are MIT [V]. MeanFlow is stronger on ImageNet [V] but needs a JVP (memory, and torch.func.jvp caveats with DDP [V]). Adding an optional condition input later merges this with Rank 2.
- **Expected gain [E]:** better range than the GAN (twin rate down), similar or worse art-likeness at first, modest composition gain.
- **Fails if:** one-step quality at 5k images collapses to blur (one-step models are usually reported at ImageNet scale; no small-data results were found); or it memorises, as in Rank 2.
- **Data: needs more data** for the same reasons as Rank 2, though less critically, because it is unconditional. **Move the bigger-data step earlier.**

### Rank 4 (cheap side bet): Projected GAN / FastGAN with a self-supervised feature network (G3 with DINO v1 or MAE in place of EfficientNet)
- **Why:** best documented small-data GAN on art: Art Painting, 1,000 images, FID 27.96 vs 43.07 for StyleGAN2-ADA [V]; fast (hours, not days, in the paper [V]); MIT code [V].
- **Text caveat:** the default EfficientNet-Lite1 is ImageNet-label-supervised [V] (allowed by the letter of the rule, but a human-label prior). The paper tried R50-CLIP, which is **text** and must not be used [V]. The paper did not test DINO or MAE feature networks [V], so the swap is unverified.
- **Fails if:** the feature network lets the generator please the discriminator in feature space while the pixels drift (a known projected-GAN artefact risk [M]); and **never evaluate it with the same encoder it was trained against** (Section 0, item 4). Using DINOv2 in the discriminator would tie the project to DINOv2 space again, which the owner wants to avoid; use DINO v1 or MAE.
- **Data:** works on current data.

### Not recommended now
- StyleGAN-XL (needs class labels and an ImageNet classifier; scale beyond the budget).
- RAE / MeanFlow-RAE (2511.13019 [V]) in DINOv2 space: strong at ImageNet scale, but it ties the generator to DINOv2 space, which the owner wants to leave. Reconsider with MAE as the encoder after the bigger-data step.
- pMF from scratch: paper-scale compute; LPIPS / ConvNeXt-V2 perceptual losses (label-level) [V].
- Drifting models: conceptually close to the project's feature-space distribution terms, but reported with effective batches of 4,096 to 8,192 [V]. Revisit when code exists and small-batch behaviour is known.
- Token models (MaskGIT, MAR) and TarFlow: no cheap differentiable path.
- Anything built on Stable Diffusion, SDXL, FLUX or PixArt denoisers (text). The SD VAE alone (D3b) is a judgement call for the owner (filter-level contamination).

### Which options truly need more and better data
- **Need it:** Rank 2 (conditional decoder), Rank 3 (one-step flow from scratch), and every from-scratch diffusion model (D1, D2, D4, D7, F1, O1, O2). At 4.7k images they are expected to memorise (Section 0).
- **Work on current data:** Rank 1 (R3GAN / StyleGAN2-ADA), Rank 4 (Projected / FastGAN), and the fine-tuning of public non-text checkpoints (with the label-prior caveat).

---

## 4. Uncertainties and things not verified

- No paper found reports diffusion, flow or one-step models trained on ~5k abstract paintings at 128 px. All expected gains in Section 3 are my estimates [E].
- The memorisation thresholds come from CelebA / LSUN at 80 px (2310.02557) and medical images (2305.07644); abstract art has lower semantic complexity and may generalise earlier or later. Not known.
- R3GAN at batch 64 with DiffAugment, and R3GAN at 128 px: not reported in the README configurations [V]; the paper may contain more [not read in full].
- Whether R3GAN reduces the twin rate on this data: plausible from its stated aim, unproven.
- Projected GAN with DINO v1 / MAE feature networks: not tested in the paper [V]; performance unknown.
- Licence of Gsunshine/py-meanflow and of JiT / RAE / Drifting code: not read or not found.
- T4 lacking native bf16, and fp16 being stable for these models on T4: from memory [M]; validate with a short run.
- Kaggle quota accounting for 2 x T4 sessions (whether the hours count once or twice): not checked.
- LAION-Aesthetics being selected by a CLIP-based predictor (the SD VAE's filter-level contamination): from memory [M].
- Diff-AE details (z_sem dimension, deterministic DDIM decoding): the abstract confirms a two-part code with near-exact reconstruction [V]; the rest is [M].
- RCG's encoder is MoCo v3 ViT-B/L ([V], repo); the 256-d representation size is [M].
- Compute estimates in Section 0 and the table are rough conversions (T4 ≈ 0.5 V100 at fp16) [E].

## Sources (all checked this session unless marked)

arXiv: 2501.05441 (R3GAN); 2006.06676 (StyleGAN2-ADA); 2006.10738 (DiffAugment); 2111.01007 (Projected GAN, also via ar5iv HTML); 2101.04775 (FastGAN); 2202.00273 (StyleGAN-XL, via ar5iv HTML); 2112.09130 (Vision-aided GAN); 1903.05628 (mode-seeking GAN); 2206.00364 (EDM); 2010.02502 (DDIM); 2112.10752 (LDM); 2212.09748 (DiT); 2511.13720 (JiT); 2304.12526 (Patch Diffusion); 2510.11690 (RAE, HTML); 2511.13019 (MeanFlow-RAE); 2312.03701 (RCG); 2111.15640 (Diff-AE); 2210.06462 (Self-guided diffusion); 2410.06940 (REPA); 2210.02747 (Flow matching); 2209.03003 (Rectified flow); 2505.13447 (MeanFlow); 2512.02012 (Improved MeanFlow); 2601.22158 (pixel MeanFlow, HTML); 2410.12557 (Shortcut models); 2303.01469 (Consistency models); 2406.14548 (ECT); 2410.11081 (sCM); 2311.18828 (DMD); 2602.04770 (Drifting models, HTML); 2412.06329 (TarFlow); 2012.09841 (VQGAN); 2202.04200 (MaskGIT); 2406.11838 (MAR); 2309.17400 (DRaFT); 2402.14017 (D-Flow); 2307.10711 (AdjointDPM); 2409.08861 (Adjoint Matching); 2303.13703 (DOODL); 2310.02557 (generalisation vs memorisation, HTML); 2310.02664 (memorisation in diffusion); 2212.03860 (data replication); 2502.21278 (ambient diffusion and memorisation); 2305.19256 (Ambient Diffusion); 2305.07644 (diffusion vs GAN memorisation); 2104.14294 (DINO); 2304.07193 (DINOv2); 2111.06377 (MAE); 2104.02057 (MoCo v3).

Repositories / model cards: github.com/brownvc/R3GAN; github.com/NVlabs/stylegan2-ada-pytorch; github.com/NVlabs/edm; github.com/autonomousvision/projected-gan; github.com/autonomousvision/stylegan-xl; github.com/LTH14/rcg; github.com/openai/consistency_models; github.com/kvfrans/shortcut-models; github.com/Gsunshine/py-meanflow; huggingface.co/stabilityai/sd-vae-ft-mse.
