# Course Project Report Draft

`main.tex` is the English NeurIPS-format working draft, based on the teacher's
paper samples. It reuses the course proposal's local `neurips_2026.sty`.
The compiled report is `main.pdf`.

The primary table uses the committed five-fold result at `results/cv_cpu.json`.
The three images in `figures/` are the exact outputs shown in the report.
From the project root, rerun the experiment on CPU with:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_selector_paper.py --output docs/report/results/cv_cpu.json
```

This command requires the local candidate bank, rankings, and cached CLIP
features under `experiments/selector100_t15_v14_20261028/`. The committed
result supports inspection and report compilation on another computer.

From `docs/report`, compile the paper with Tectonic:

```bash
tectonic -X compile main.tex --outdir .
```

The evaluation uses five fixed source folds from
`cross_validation_balanced/report.json` in the same experiment directory.
Each fold holds out 20 sources. Of the remaining 80, the next outer fold's
20 sources select the acceptance threshold after training on the other 60.
The final model is trained afresh on all 80 sources. Both fits use 80 epochs
and seed `2012 + fold` (folds 1–5). Threshold candidates are
`[-inf, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]`.
Ties favor the finite threshold closest to zero, then the earlier list entry.
Feature-shuffling results use 30 permutations per outer fold with the same
fold seed. The script loads cached CLIP vectors and uses four CPU threads.

The paper calls `action_only` the **recipe baseline**: its image embeddings
are zeroed, while recipe descriptors and estimated subject count remain.
`pre_generation` is the pre-render selector; `full` is the post-render scorer.
The recorded comparisons use the same completed candidates in each fold.
The fixed-position baseline chooses c06 in every training fold; c06 refers
to a candidate slot whose recipe varies across sources.

The report remains a draft. Its 100 sources informed model development,
and its labels reflect one reviewer. Recorded times estimate historical
generation work. Fresh sources, additional reviewers, and a timed editing
pipeline would strengthen the evidence for generalization and compute savings.
The figure assets and the full five-fold result are included with the report.
