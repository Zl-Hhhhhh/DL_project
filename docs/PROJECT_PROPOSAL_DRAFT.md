# Object-Aware Preference Learning for Controllable Image Collage

## 1. Motivation and Research Questions

Controllable collage editing produces multiple plausible arrangements of the same subject. Selecting among them requires considering both the source image and an editing goal. Neural Collage Transfer studies material-based reconstruction [1], while composition and style attributes have been studied for aesthetic assessment [2]. This project instead investigates preference learning among alternative edits of the same image under limited annotation. **Do explicit source-to-edit subject features improve ranking beyond global image features? Do geometric and color features improve annotation efficiency?** The contribution sought is an empirical account of when these representations help, rather than a new image generator or a universal aesthetic metric.

## 2. Proposed Method

I will extend an existing segmentation and rendering pipeline to generate eight poster-style candidates per source image, varying subject position, scale, spacing, and color treatment. Each source will receive one predefined editing brief specifying a composition and color goal. Candidates will preserve the extracted subject geometry and use controlled backgrounds. A frozen DINOv2 encoder [3] will extract global source/candidate embeddings and masked subject-crop embeddings on a shared neutral background. A small multilayer perceptron will predict a goal-conditioned preference score using these embeddings and explicit features, including distance to the target position, subject area, and foreground/background color contrast. Only the ranking head will be trained, using pairwise logistic loss. Cached features make this practical on the available 8 GB GPU.

## 3. Experimental Plan

I plan to collect 120 additional source images, with 80/20/20 training/validation/test splits grouped by source and near-duplicate identity. Existing development images will be excluded. For each source, I will annotate eight candidate pairs with randomized presentation order, recording ties and repeating a subset to estimate consistency. These are author preferences, not population-level aesthetic judgments.

All methods will share candidate pools, editing briefs, and evaluation pairs. Baselines will include random selection, a fixed composition rule, and a global-feature ranker. A factorial ablation will add subject features, design features, or both to the global ranker, using the same ranking-head architecture with omitted feature blocks zeroed. Training budgets of two, four, and eight labeled pairs per training image will test annotation efficiency. Evaluation will report held-out non-tied pairwise agreement, tie rates, blind top-choice comparisons, and uncertainty estimated by resampling source images. Three training seeds will assess optimization variability. Failure analysis will examine segmentation errors and differences between object categories.

## 4. Feasibility and Expected Outcomes

The current prototype supports segmentation, composition controls, foreground color preservation, and reproducible rendering. On 12 development images, eight passed segmentation and geometric checks, producing 64 replay-verified candidates. These checks do not establish aesthetic quality; learned ranking and preference evaluation remain unimplemented. The planned outcome is a reproducible study of representation choice and label efficiency, including negative results and limitations of single-annotator supervision. I will finalize data and annotations in October, conduct training and ablations in November, and complete the report and presentation before December 11.

## References

[1] G. Lee et al. [Neural Collage Transfer: Artistic Reconstruction via Material Manipulation](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Neural_Collage_Transfer_Artistic_Reconstruction_via_Material_Manipulation_ICCV_2023_paper.html). ICCV, 2023.

[2] L. Celona et al. [Composition and Style Attributes Guided Image Aesthetic Assessment](https://arxiv.org/abs/2111.04647). IEEE Transactions on Image Processing, 31:5009-5024, 2022.

[3] M. Oquab et al. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193). arXiv:2304.07193, 2023.
