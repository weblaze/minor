# Human learnability test (proposal, 2026-11-22)

Status: proposed design, not decided and not built. Nothing here runs until the project owner approves it.

## Question

The machine-side checks show that a learned reader can name a test track from its image: S3 78 % top1 by its own reader, 52 % by a fresh reader on DINO v1 (run log 2026-11-22). They do not show that a person can learn to read the images. This test asks: after seeing a small number of example pairs, can a person tell which image belongs to a sound they have not heard before, better than chance, and better than with the controls?

This differs from the human judging already piloted (2026-11-19), which asks which of several images fits a sound, without any learning phase. Here people learn first, then apply what they learnt to new tracks. No text, labels or explanations are shown at any point, so the only source of knowledge is the pairs themselves.

## Design

One session per participant, about 20 minutes, in the browser (same clip set and hosting as the judging page).

1. Learning phase (about 8 minutes). 24 training tracks, each shown as a 10 s clip with its image. The participant plays each pair as often as they like. No questions asked.
2. Test phase (about 10 minutes). 24 test tracks never seen in learning. For each: play the clip, choose its image among 4 (the true image and 3 images of other test tracks, drawn at random). No feedback during the test.
3. Second learning and test block (optional, if time allows): a further 24 learning pairs, then 24 new test trials. An increase from block 1 to block 2 is the clearest sign of learning.

Trials per condition per participant: 24 (or 48 with block 3). Chance: 25 %.

## Conditions (between participants)

Each participant learns one system, so what they learn about one code does not carry over into another.

| Condition | Role |
|---|---|
| S3 + GAN (s3) | Main system: the invented picture language |
| S3 + GAN (random_m) | Control: the same generator and pipeline, with an untrained mapper |
| S3 + GAN (shuffled) | Control: trained, but with pairs shuffled so no audio information is in the code |
| Q1 + GAN or S2 anchored | Comparison: a structure-based system (similar sounds give similar images) |

The shortlist decides the comparison condition. The random_m and shuffled conditions are needed. Without them a score above 25 % could come from people picking the image that "looks musical" for any sound.

## Measures

- Test accuracy per participant against chance (one-sided binomial), and by condition (mixed logistic regression: participant and track as random effects).
- Learning: accuracy of block 2 minus block 1.
- Comparison with the machine: the same 4-way trials scored by the fresh DINO v1 reader (and by CAV-MAE), so that human and machine accuracy are on the same scale.
- Optional, after the test: one free-text answer ("what did you notice?"). It is collected only to report it, and is never used to fit or select anything.

## Size

With 24 trials per participant, a mean accuracy of 35 % against 25 % chance needs about 15 participants per condition (power 0.8, alpha 0.05, allowing for spread between participants). Four conditions: about 60 participants. A smaller pilot (5 per condition) would show whether the effect is near that size before recruiting more.

## Risks

- 10 s clips may be too short to hear what the code uses; the window test (S3 window top1 0.40) shows that one window carries less than the whole track.
- 24 pairs may be too few to learn a code that the machine learnt from 6408 tracks. Block 2 measures this.
- If the comparison condition gives higher accuracy, then structure similarity is easier for people to learn than an invented code. That result is also useful.

## Needed before building

- The shortlist (which systems).
- Clip and image sets for the chosen systems: the existing `abstraction-judging-clips` set covers the test tracks; images come from the saved evaluation images of each run.
- A decision on the participants: who, how many, and whether the pilot runs first.
