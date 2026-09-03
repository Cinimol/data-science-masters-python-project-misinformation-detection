# Consistency-Gated Multimodal Misinformation Detection

Artefact for **UFCF9Y-60-M CSCT Masters Project** (MSc Data Science, UWE Bristol).
*Multimodal Detection of Misinformation Using Deep Learning and Large Language Models*.

The artefact does three things:

1. **Builds** a 21,798-post multimodal corpus (headline + image + platform
   metadata) from the public Fakeddit benchmark.
2. **Demonstrates a shortcut** in how Fakeddit is normally evaluated: because
   each label is assigned by the *subreddit a post came from*, a model that sees
   only the subreddit name and no content at all scores near-perfectly under the
   random split used throughout the published literature. Two corrected
   protocols (source-disjoint and temporal) are supplied alongside it.
3. **Proposes and evaluates CGF** (Consistency-Gated Fusion), a multimodal
   detector that treats the *image-headline relationship* as a first-class
   input rather than leaving the classifier to infer it, and compares it against
   thirteen baselines including a zero-shot LLM judge, under all three protocols,
   with confidence intervals, significance tests, ablations and behavioural
   robustness probes.

Everything runs on a laptop CPU. No GPU, no paid API, no manual downloads.

---

## 1. Quick start (macOS, Windows or Linux)

### 1.1 Install Python

Requires **Python 3.10 to 3.12**.

**macOS**: check what you have, then install if needed:

```bash
python3 --version
# if missing or older than 3.10:
brew install python@3.11        # https://brew.sh
```

If you do not have Homebrew, download the installer from
<https://www.python.org/downloads/macos/>.

**Windows**: install from <https://www.python.org/downloads/windows/> and tick
*"Add python.exe to PATH"* during setup.

### 1.2 Get the code and create an isolated environment

```bash
git clone <YOUR-REPOSITORY-URL> mmid
cd mmid

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins CPU-only PyTorch, so nothing tries to install CUDA.
On an Apple-silicon Mac the code automatically uses the **MPS** (Metal) backend,
which is roughly 3 to 5 times faster than CPU for the encoder stages; `src/config.py`
falls back to CPU on Intel Macs and to CUDA if a GPU is ever available. To force
CPU for a reproducibility check: `export FORCE_CPU=1`.

### 1.3 The easy way: one command

```bash
python run.py
```

This prints a menu, works out which stages have already produced output, greys
out anything whose prerequisites are missing (and says which option supplies
them), and runs whichever you pick. It is the recommended entry point; the
stage-by-stage commands below remain available and do exactly the same work.

In VS Code you can also press the Run button with `run.py` open, or pick
**▶ START HERE (menu)** from the Run and Debug dropdown.

### 1.4 Run the whole pipeline

```bash
make all
```

or, equivalently, stage by stage:

```bash
python -m src.build_dataset          # corpus and the three split protocols
python -m src.features --stages base # frozen CLIP and DistilRoBERTa
python -m src.features --stages llm  # language-model states (optional)
python -m src.train                  # the 165-run experiment grid
python -m src.robustness             # ablation, behavioural probe, cost
python -m src.analyse                # metrics, significance, figures
python -m src.extra_analyses         # rotation, probe null, ranking, Holm
python -m src.triage                 # the reviewer triage queue
```

Measured runtimes for every stage are in Appendix C of the report, which is the
single source for them.

The written report is rebuilt from Markdown with `make report`, which runs
pandoc and then applies the table design in `src/style_docx.py`.

Every figure is also implemented in MATLAB. Run `make matlab-data` to export
the plotting data, then `run_all_figures` from the `matlab/` directory; see
`matlab/README.md`.

Stage outputs are cached, so re-running skips completed work. If you want a
five-minute smoke test of the full path before committing to the real run:

```bash
make smoke
```

### 1.4 What you get

| Path | Contents |
|---|---|
| `data/processed/corpus.parquet` | the corpus manifest with all three split assignments |
| `data/processed/feat_base.npz` | frozen CLIP + DistilRoBERTa representations |
| `data/processed/feat_llm.npz` | LLM hidden states and zero-shot P(fake) |
| `results/runs.csv` | one row per (protocol, model, seed) |
| `results/confidence_intervals.csv` | seed-averaged metrics with bootstrap CIs |
| `results/significance.csv` | McNemar and paired-bootstrap comparisons |
| `results/robustness.csv`, `results/behavioural.csv` | perturbation and behavioural probes |
| `figures/*.png` | every figure in the report |
| `matlab/data/*.csv` | the same figure data, flattened for the MATLAB suite |
| `figures_matlab/*.png` | the figures as redrawn by the MATLAB scripts |
| `diagrams/*.drawio` | editable draw.io sources for the flowcharts |

---

## 2. Data provenance and licensing

The corpus is derived from **Fakeddit** (Nakamura, Levy and Wang, 2020), a
public research benchmark of Reddit submissions released for academic use. This
project uses:

* the metadata table, from a public mirror of the 100k multimodal split,
  `https://huggingface.co/datasets/rtfarchitect/fakeddit_sample`;
* the associated Reddit preview images, fetched from the URLs in that table.

No accounts are scraped, no private data is collected and no personal data is
redistributed: `src/build_dataset.py` **drops the `author` column** from every
saved artefact and retains only a binary `has_author` flag. Images are cached
locally for reproducibility and are not redistributed with this repository.
About 4% of preview URLs have expired; the manifest records exactly which rows
survived, so a re-run on a later date is auditable rather than silently
different.

See Chapter 5 of the report for the full ethical analysis.

---

## 3. Repository layout

```
src/
  config.py         all paths, hyper-parameters, device selection
  build_dataset.py  Stage 1: sampling, image download, the three split protocols
  features.py       Stage 2: frozen CLIP / DistilRoBERTa / LLM representations
  models.py         Stage 3: MLP baselines, late fusion, and CGF (proposed)
  metrics.py        Stage 4: metrics, bootstrap CIs, McNemar, calibration
  train.py          Stage 4: the experiment grid
  robustness.py     Stage 6: modality ablation, out-of-context probe, cost
  analyse.py        Stage 5: aggregation, significance, all figures
  finetune_text.py  Stage 7: end-to-end fine-tuned text baseline (optional)
  style_docx.py     Stage 8: applies the report's table design to the .docx
  export_matlab.py  Stage 9: flattens the figure data to CSV for MATLAB
  extra_analyses.py Stage 10: rotation, probe null, ranking, prior, Holm
  fig_architecture.py Stage 11: the system and model architecture figure
  triage.py         Stage 12: the reviewer-facing ranked queue
  flowcharts.py     Stage 13: the flowcharts, as .drawio and as figures
  make_tables.py    renders the Chapter 6 tables from the results files
  predict.py        scores one image/headline pair (Section 5 above)
matlab/
  *.m               a MATLAB implementation of all eight report figures
diagrams/
  *.drawio          editable draw.io sources for every flowchart
tests/
  test_pipeline.py  25 unit tests: leakage, split disjointness, minimisation, metrics, shapes
```

Run the tests with:

```bash
python -m pytest tests/ -q          # 25 passed
```

---

## 4. Reproducing individual results

```bash
# just the source-disjoint protocol, just the proposed model
python -m src.train --protocols source --models cgf,cgf_llm

# only the shortcut demonstration (Section 6.2 of the report)
python -m src.train --protocols random,source --models subreddit_probe

# regenerate every figure from cached predictions, without retraining
python -m src.analyse
```

Seeds are fixed in `src/config.py` (`SEEDS = [13, 42, 7]`). Every reported
number is the mean over those three seeds; spread is reported as a standard
deviation across seeds and as a percentile bootstrap CI over test items.

Exact reproduction of the corpus depends on which preview URLs are still live
on the day you run it. To reproduce the *reported* numbers exactly, use the
cached `feat_base.npz` / `feat_llm.npz` bundles rather than rebuilding.

---

## 5. Scoring one image and headline

Every stage above evaluates the artefact in aggregate, over a held-out test
split. `src/predict.py` does the complementary thing a demo or a viva actually
needs: point it at one image and one headline and get a verdict, using the
same frozen encoders and the same Consistency-Gated Fusion model as the rest
of the study.

```bash
python -m src.predict --image path/to/photo.jpg \
                      --headline "Scientists discover water on the surface of the sun"
```

The first call trains and caches a small checkpoint (a couple of minutes on
CPU, cached under `results/checkpoints/`); every call after that answers in
seconds. By default it uses the report's proposed system, CGF + LLM +
adversarial, if `data/processed/feat_llm.npz` has been exported, and falls
back to plain CGF otherwise; pass `--variant cgf` (or any other row of Table 1
that is a CGF variant) to choose explicitly.

The output states a verdict, the probability behind it, the CLIP
image/headline agreement the model saw, and the learned gate's activation --
useful in a demo for showing *why* the model reached that verdict, not only
what it decided.

Five of the model's nine metadata features (post score, comment count, upvote
ratio, whether the author field was present, whether it was a self-post)
describe platform behaviour that does not exist for an image and headline
typed in from outside Reddit; unset, they default to the training corpus's
median values. If you are checking an actual Reddit post and know the real
figures, supply them for a more faithful score:

```bash
python -m src.predict --image photo.jpg --headline "..." \
                      --score 340 --comments 58 --upvote-ratio 0.71
```

One point worth being direct about: this reproduces Fakeddit's own definition
of "fake" -- a headline claim the image does not corroborate, as the
benchmark's annotators judged it -- not a check for AI-generated or digitally
altered pixels. Section 6.2 of the report is the reason that distinction
matters: the benchmark's labels are largely a function of which subreddit a
post came from, so this script is a demonstration of the artefact's proposed
mechanism, not a general-purpose fake-image detector.

---

## 6. Attribution

Third-party components, all used under their own licences:

| Component | Source | Licence |
|---|---|---|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` (Radford et al., 2021) | MIT |
| DistilRoBERTa | `distilroberta-base` (Sanh et al., 2019) | Apache-2.0 |
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` (Qwen Team, 2024) | Apache-2.0 |
| Fakeddit | Nakamura, Levy and Wang (2020) | research use |
| PyTorch, Transformers, scikit-learn, pandas, matplotlib | Open source | BSD / Apache-2.0 |

All code in `src/` and `tests/` was written for this project. Where a standard
technique is implemented directly rather than imported (McNemar's test, expected
calibration error, the percentile bootstrap), the docstring names the method so
the source of the idea is clear.
