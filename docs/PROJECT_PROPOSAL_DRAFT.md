# Photo Editing with Controllable Styles, Filters, and Composition

## 1. Project Goal

I want to build a photo editor that turns an input image into a chosen style, filter, and composition while keeping its main subject recognizable. Example looks include warm film-like colors, muted vintage tones, and collage posters with off-center subjects. I will choose a few online reference looks and document their sources and visible popularity signals.

## 2. How It Will Work

Segmentation will separate the subject from the background. Cropping or repositioning will adjust subject placement and empty space; color filters will change the overall mood. Reference-guided styling [1] will provide a separate option. Composition and color cues [2] will guide a few output candidates. A small model using frozen visual features [3] will learn to select the candidate that best fits the requested look. This selection step supports the editing tool.

## 3. Testing the Results

I plan 120 new photos, split 80/20/20 for training/validation/testing by source. For each editing goal, I will compare fixed presets, subject-aware processing, and learned selection on shared candidates. Blinded author comparisons will assess whether outputs match the requested look, preserve subject details, and avoid visible artifacts. I will also measure processing time and examine failures such as poor masks. Online popularity guides reference selection, not the quality score.

## 4. Current Progress and Schedule

The prototype supports subject segmentation, composition controls, and foreground color preservation. Eight of 12 development images passed segmentation/geometric checks, producing 64 replay-verified candidates; this does not establish visual quality. Reference curation and data preparation are planned for October, comparisons and model training for November, and final reporting for early December.

## References

[1] H. Wang et al. [InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation](https://arxiv.org/abs/2404.02733). arXiv:2404.02733, 2024.

[2] L. Celona et al. [Composition and Style Attributes Guided Image Aesthetic Assessment](https://arxiv.org/abs/2111.04647). IEEE Transactions on Image Processing, 31:5009-5024, 2022.

[3] M. Oquab et al. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193). arXiv:2304.07193, 2023.
