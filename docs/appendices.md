# Appendices

## Appendix A: Record of meetings with supervisor

Supervision was fortnightly. The record below is completed by the student from
the meeting log; each row names the decision taken, because a record of
attendance without decisions evidences contact rather than management.

**Table A1:** Record of supervision meetings.

| # | Date | Phase | Items discussed | Decision or action agreed | Supporting evidence |
|:--|:-----|:------|:----------------|:--------------------------|:--------------------|
| 1 |  | 1 | Topic scoping, research question |  | A1 |
| 2 |  | 1 | Literature scope, benchmark choice |  | A2 |
| 3 |  | 1 | Proposal and ethics form |  | B |
| 4 |  | 2 | Corpus design, sampling cap |  | A3 |
| 5 |  | 2 | Shortcut probe result; change of direction |  | K, SC-01 |
| 6 |  | 3 | Protocol B design, baselines |  | A4 |
| 7 |  | 3 | Ablation results, adversarial variant |  | A5 |
| 8 |  | 4 | Rotation result and its effect on the claim |  | A6 |
| 9 |  | 4 | Draft chapters, evaluation framing |  | A7 |

*Instruction to the student: enter the dates, discussion notes and agreed
actions from the supervision log, and attach the referenced evidence items.
This table must be completed before submission.*

## Appendix B: Ethical approval

**Table B1:** Ethics review record.

| Field | Entry |
|:------|:------|
| Reviewing body | Module ethics process, UWE Bristol |
| Category | Secondary analysis of pre-existing public data; no human participants |
| Personal data processed | Pseudonymous Reddit identifiers, dropped at build time |
| Lawful basis | Legitimate interests, read with the UK GDPR research provisions |
| Approval reference |  |
| Date of approval |  |
| Approving supervisor |  |

*Instruction to the student: insert the completed approval form issued by the
supervisor after this table, and enter the reference and date above. This is a
mandatory submission item.*

## Appendix C: Artefact access and replication guide

**Repository:** [repository URL]

### C.1 Requirements

Python 3.10 to 3.12; approximately 4 GB of free disk space for images, cached
representations and model weights; and an internet connection for the first run
only. No GPU is required. On an Apple-silicon Mac the code selects the Metal
Performance Shaders backend automatically and falls back to CPU elsewhere.
Setting `FORCE_CPU=1` pins execution to CPU for a reproducibility check.

### C.2 Installation

```bash
git clone <repository-url> mmid
cd mmid
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest tests/ -q         # expected result: 25 passed
```

### C.3 Reproducing the results reported in Chapter 6

The pipeline is invoked with `make all`, or stage by stage as below. Runtimes
were measured on the two-core CPU used for this study.

**Table C1:** Pipeline stages and measured runtimes.

| Command | Purpose | Runtime |
|:--------|:--------|--------:|
| `python -m src.build_dataset` | Download metadata, sample corpus, fetch 21,798 images | 13 min |
| `python -m src.features --stages base` | Frozen CLIP and DistilRoBERTa representations | 32 min |
| `python -m src.features --stages llm` | Language-model hidden states and zero-shot judgements | 100 min |
| `python -m src.train` | 21 systems, 3 protocols, 3 seeds for the 18 trainable systems (165 runs) | 40 min |
| `python -m src.robustness` | Modality ablation, out-of-context probe, perturbation, cost | 35 min |
| `python -m src.analyse` | Tables, significance tests, figures | 2 min |
| `python -m src.extra_analyses` | Probe null, per-community, ranking, prior decomposition, Holm, rotation | 55 min |
| `python -m src.fig_architecture` | Figure 1 | 5 s |
| `python -m src.triage` | Reviewer triage queue (Appendix P) | 5 s |
| `python -m src.export_matlab` | Flatten the figure data for the MATLAB suite | 5 s |
| `python -m src.finetune_text --protocol source` | End-to-end fine-tuned control (Section 6.10) | 26 min |

Apple-silicon MPS is roughly three to five times faster on the two encoder
stages. A five-minute end-to-end smoke test on a 400-post sample is available
as `make smoke`.

### C.4 Output locations

**Table C2:** Artefact outputs and their correspondence to the report.

| Path | Contents |
|:-----|:---------|
| `data/processed/corpus.parquet` | Corpus manifest with all three split assignments |
| `data/processed/feat_base.npz`, `feat_llm.npz` | Cached frozen representations |
| `results/runs.csv` | One row per protocol, model and seed |
| `results/confidence_intervals.csv` | Table 1 and Appendix O |
| `results/significance_holm.csv` | Corrected significance tests (Appendix O) |
| `results/loco.csv` | Leave-one-community-out rotation (Table 2, Appendix O) |
| `results/per_community.csv`, `ranking.csv` | Section 6.9 |
| `results/extra_analyses.json` | Probe null and prior decomposition (Sections 6.2, 6.3) |
| `results/triage_queue.csv` | Appendix P |
| `figures/`, `figures_matlab/` | Every figure, in both implementations |
| `diagrams/*.drawio` | Editable sources for the flowcharts (Appendix T) |

### C.5 MATLAB implementation of the figures

Every figure is implemented a second time in MATLAB, in `matlab/`, so the
visual results can be inspected without a Python environment. After
`python -m src.export_matlab` has written the plotting data:

```matlab
cd matlab
run_all_figures
```

Fourteen files are written to `figures_matlab/`. Individual scripts may be
called directly, and those that vary by protocol accept it as an argument, for
example `fig4_calibration('source')`. Running `mmid_check` first reports the
interpreter and release in use, reads every expected CSV with its row and
column counts, and prints the command that fixes any failure it finds.

The scripts require core MATLAB only: no toolbox is used, the reader is built
on `fileread`, `regexp` and `sscanf` rather than `readtable` or `textscan`, and
figures are written with `print` rather than `exportgraphics`. Three points
where MATLAB and Octave genuinely differ are resolved centrally in
`mmid_style.m` by querying the interpreter rather than assuming one of them:
handle-array preallocation (`gobjects` against `zeros`, without which MATLAB
R2014b and later reject a bar handle assigned into a numeric array), legend
property assignment after creation rather than as arguments, and escaping of
underscores in tick labels on backends that accept `TickLabelInterpreter` and
then ignore it. The suite was executed end to end under GNU Octave 8.4 as a
portability check and writes all fourteen figures in under twenty seconds.

`src/analyse.py` remains authoritative, since it produces the figures embedded
in this report; `src/export_matlab.py` flattens cached arrays rather than
recomputing any quantity, so the two implementations cannot disagree. As an
independent check, the gate against consistency correlation of Figure 8 is
computed separately on each side and agrees to machine precision.

### C.6 Note on exact reproducibility

The corpus depends on which Reddit preview URLs remain live on the day the
build is run. At build time 3.8% had expired, and that proportion will grow.
All model training is seeded and deterministic given a fixed corpus, so exact
reproduction of the reported numbers requires the cached `feat_base.npz` and
`feat_llm.npz` bundles supplied with the repository release rather than a
rebuild of the corpus from scratch.

## Appendix D: Third-party components, licences and selection rationale

**Table D1:** Third-party components, licences and why each was selected.

| Component | Source | Licence | Use and deciding reason |
|:----------|:-------|:--------|:------------------------|
| Fakeddit | Nakamura, Levy and Wang (2020) | Research use | Corpus. The only public multimodal misinformation benchmark large enough to hold out whole communities |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` (Radford *et al.*, 2021) | MIT | Image encoder and consistency scorer. Its joint image-text space yields a direct agreement measure a vision-only encoder cannot produce |
| DistilRoBERTa | `distilroberta-base` | Apache-2.0 | Text encoder. 40% smaller than RoBERTa-base at comparable quality, and per-item cost is requirement N4 |
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` (Qwen Team, 2024) | Apache-2.0 | Language-model judge and encoder. Open weights and CPU-runnable, so no API key is required to replicate |
| PyTorch | pytorch.org | BSD-3-Clause | Model implementation. MPS support keeps the study laptop-runnable |
| Transformers, Tokenizers | Hugging Face | Apache-2.0 | Encoder loading and tokenisation |
| scikit-learn, SciPy | scikit-learn.org, scipy.org | BSD-3-Clause | Non-neural baselines, metrics and statistical tests; reference implementations reduce metric-error risk |
| pandas, NumPy, PyArrow | Open source | BSD-3-Clause / Apache-2.0 | Data handling and cached artefacts |
| matplotlib | matplotlib.org | PSF-based | Figures |
| pytest | pytest.org | MIT | Test-driven guarding of leakage and correctness |

All code in `src/`, `tests/` and `matlab/` was written for this project. Where a
standard method is implemented directly rather than imported, the docstring
names the method and its source: McNemar's exact test (McNemar, 1947;
Dietterich, 1998), expected calibration error (Guo *et al.*, 2017), the
percentile bootstrap (Efron and Tibshirani, 1993), Holm's step-down procedure
(Holm, 1979) and the gradient-reversal layer (Ganin *et al.*, 2016).

## Appendix E: Test suite

Twenty-five tests run in about six seconds and were used as a regression gate
before every result reported in Chapter 6. They target the failure modes that
would invalidate a result silently rather than crash the run.

**Table E1:** Test coverage by failure mode.

| Failure mode guarded | Tests | Test names | Why it would otherwise be silent |
|:---------------------|------:|:-----------|:---------------------------------|
| Metric correctness and label polarity | 3 | `test_metrics_against_hand_computed_values`, `test_perfect_and_inverted_predictions`, `test_ece_is_zero_for_a_perfectly_calibrated_predictor` | A wrong macro-F1 or an inverted label is consistent across systems and therefore invisible in a comparison |
| Significance testing on known inputs | 2 | `test_mcnemar_is_symmetric_and_significant_when_it_should_be`, `test_mcnemar_identical_systems_are_not_significant` | An incorrect test would produce plausible p-values |
| Split disjointness and coverage | 3 | `test_source_disjoint_split_shares_no_subreddit_between_train_and_test`, `test_temporal_split_is_ordered_in_time`, `test_random_split_covers_every_row_exactly_once` | A leaked community would inflate the corrected protocol and destroy the central finding |
| No leakage in the feature scaler | 1 | `test_standardiser_uses_training_statistics_only` | Test-set statistics leaking into scaling inflates every score without raising an error |
| Data minimisation (NF3) | 1 | `test_no_author_column_survives_the_build` | A regression would silently redistribute re-identifiable pseudonyms while every model still trained |
| Metadata integrity | 1 | `test_derived_metadata_columns_are_finite` | A NaN or infinity would propagate into standardisation |
| Model shape contracts | 8 | `test_cgf_forward_shapes` (5 parametrisations), `test_cgf_with_llm_stream`, `test_mlp_and_late_fusion_shapes`, `test_model_is_deterministic_in_eval_mode` | A mis-shaped tensor usually broadcasts rather than raising |
| Gate bounds and recording | 1 | `test_gate_is_bounded_and_recorded` | An unbounded gate would invalidate the interpretability claim of F3 |
| Text perturbation behaviour | 5 | `test_perturbations_return_non_empty_strings` (4 parametrisations), `test_lowercase_perturbation_removes_capitals` | An inert perturbation would make the robustness result meaningless |

## Appendix F: Corpus composition

**Table F1:** Composition by source community and Protocol B role.

| Community | Posts | Label | Protocol B role |
|:---|---:|:---|:---|
| r/upliftingnews | 1598 | true | val |
| r/savedyouaclick | 1598 | false | test |
| r/nottheonion | 1596 | true | test |
| r/neutralnews | 1579 | true | train |
| r/propagandaposters | 1561 | false | train |
| r/subredditsimulator | 1555 | false | train |
| r/photoshopbattles | 1535 | true | train |
| r/misleadingthumbnails | 1524 | false | val |
| r/pareidolia | 1521 | false | train |
| r/fakehistoryporn | 1438 | false | train |
| r/mildlyinteresting | 1380 | true | train |
| r/usnews | 1261 | true | test |
| r/usanews | 1228 | true | test |
| r/confusing_perspective | 1187 | false | test |
| r/pic | 1128 | true | train |
| r/subsimulatorgpt2 | 103 | false | train |
| r/fakefacts | 6 | false | train |
| **Total** | **21798** | 11,305 true / 10,493 false | |

Two communities, r/subsimulatorgpt2 (103 posts) and r/fakefacts (6 posts), are
retained in training but are too small to serve as a rotation fold, which is
why Section 6.4 rotates over fifteen of the seventeen communities.

## Appendix G: Requirements traceability

**Table G1:** Requirements, acceptance criteria and verification status.

| # | Requirement | Acceptance criterion | Verified by | Status |
|:--|:------------|:---------------------|:------------|:-------|
| F1 | Classify from headline, image and metadata | All three streams present in the fused input; zeroing any one changes the output | `test_cgf_forward_shapes`; Section 6.8 | Met |
| F2 | Output a calibrated probability, not a hard label | Continuous output in [0,1]; ECE reported and below 0.15 for the proposed system | ECE 0.138, Section 6.6 | Met |
| F3 | Expose an inspectable per-item cross-modal signal | Gate readable at inference for every item; correlation with CLIP consistency reported | Figure 8; *r* = 0.19 | Met, but weakly informative |
| F4 | Produce a ranked review queue | Queue with per-row probability, band precision and evidence | `src/triage.py`; Appendix P | Met |
| NF1 | Run end to end on a laptop CPU | Full pipeline under 6 CPU-hours, no GPU | Table C1, about 5 hours | Met |
| NF2 | Reproduce exactly from a fixed seed and corpus | Two runs at the same seed give identical predictions | `test_model_is_deterministic_in_eval_mode`; identical `runs.csv` across re-runs | Met |
| NF3 | Retain no personal data | No `author` column in any written artefact | `test_no_author_column_survives_the_build` | Met |

## Appendix H: Per-category error analysis

![**Figure 9.** Per-category accuracy of the proposed system, source-disjoint split.](../figures/fig6_error_by_category_source.png)

## Appendix I: Declaration of use of generative AI

*Instruction to the student: complete this declaration in accordance with UWE
Bristol's guidance on the use of generative AI in assessed work, stating which
tools were used, for what purpose, and at which stage. This is a mandatory
submission item.*

| Field | Entry |
|:------|:------|
| Tool or tools used |  |
| Purpose |  |
| Stages at which used |  |
| Extent of use |  |
| Verification undertaken by the author |  |

## Appendix J: Evidence base for the user needs of Section 2.5

No primary study with users was undertaken, so each requirement is traced to a
documentary source and the specific extract relied on. This makes the
requirements auditable and locates precisely where the analysis would be
strengthened by interviews.

**Table J1:** Documentary evidence for each user need.

| Need | Source | Extract or finding relied on | Requirement generated | Metric reported |
|:-----|:-------|:-----------------------------|:----------------------|:----------------|
| N1 | European Union (2022), Art. 20(6) | Decisions on complaints must be "taken under the supervision of appropriately qualified staff, and not solely on the basis of automated means" | Rank for review; do not remove | Precision at k (Section 6.9) |
| N1 | Nakov *et al.* (2021) | Frames automated fact-checking as assistance to human fact-checkers rather than replacement | Triage output, human decision | Triage queue (Appendix P) |
| N2 | Zhang, Liao and Bellamy (2020) | Confidence scores improve human-AI decision accuracy only when calibrated; miscalibrated confidence degrades it | Calibrated probability, no embedded threshold | ECE (Sections 6.6, Appendix O) |
| N2 | Guo *et al.* (2017) | Modern networks are systematically over-confident; ECE is the standard measure | Report ECE alongside accuracy | ECE |
| N3 | Das *et al.* (2023) | Explanation is the dominant unmet requirement in human-centred fact-checking systems | Expose a per-item signal a reviewer can check | Gate activation, CLIP consistency (Appendix P) |
| N3 | Roberts (2019) | Documents the volume and time pressure of commercial content moderation | Evidence must be readable at a glance, not a post-hoc attribution exercise | Queue columns (Appendix P) |
| N4 | Gillespie (2018) | Platform enforcement operates at hundreds of millions of actions per period | Measure per-item inference cost as a first-class result | 427 ms vs 61 ms (Section 6.7) |
| N4 | European Union (2022), Art. 34-35 | Systemic-risk assessment applies at very large platform scale | Cost must be reported, not assumed | Figure 7 |

The principal limitation is that documentary elicitation captures stated
practice rather than tacit workflow. Section 7.3 records this, and Section 7.4
identifies interviews as the change that would most improve the analysis.

## Appendix K: Development cycles and scope decisions

**Table K1:** Iteration log. Each cycle ended in a result that determined the next.

| Cycle | Phase | Question | Result | Decision |
|:--|:--|:--------|:-------|:---------|
| 1 | 2 | Can the corpus be built reproducibly with images? | 96.2% of preview URLs resolved; manifest records the rest | Proceed; record fetch status per row |
| 2 | 2 | Does the label depend on the source community? | Content-free probe scores 1.000 under the random split | **SC-01**: reframe the project around measurement |
| 3 | 2 | Does the gap survive a corrected split? | Mean loss 0.213 macro-F1 across learned systems | Adopt Protocol B as the primary reporting protocol |
| 4 | 3 | Does explicit consistency help under Protocol B? | Ablation: removing consistency *improves* the score by 0.055 | Investigate why rather than tune |
| 5 | 3 | Why does consistency hurt across communities? | CLIP agreement varies more between communities than classes | Add the adversarial variant to penalise source information |
| 6 | 3 | Does adversarial training close the gap? | Recovers about one seventh; ECE 0.223 to 0.138 | Report as partial; do not overclaim |
| 7 | 4 | Is the protocol gap explained by class-prior shift? | Prior-matched gap 0.209 against 0.205 unmatched | Reject the alternative explanation |
| 8 | 4 | Is the architectural margin identifiable? | Fold SD 0.245 against a 0.029 model difference | **SC-02**: withdraw the architectural claim (Section 7.2) |

**Table K2:** Scope decisions.

| # | Trigger | Decision | Traded away |
|:--|:--------|:---------|:------------|
| SC-01 | Cycle 2 | Reframe from accuracy maximisation to benchmark measurement | Two planned fusion variants (cross-attention, gated bilinear) |
| SC-02 | Cycle 8 | State that the architectural comparison is not identifiable | A cleaner headline claim |
| SC-03 | Phase 3 | Freeze encoders for all systems | Peak accuracy; recovered as a control in Section 6.10 |
| SC-04 | Phase 4 | Add ranking metrics and a triage output | Time budgeted for a second corpus |

## Appendix L: Hyperparameters and training protocol

Every trainable system uses the same optimiser, schedule and capacity, so an
observed difference is attributable to the fusion mechanism rather than to a
tuning budget. The values were fixed once, before the protocol comparison, from
common practice for small MLP heads on frozen features; no per-model search was
performed, and none of the reported comparisons is therefore contaminated by
selection over hyperparameters.

**Table L1:** Shared training configuration.

| Setting | Value | Note |
|:--------|:------|:-----|
| Optimiser | AdamW | Decoupled weight decay |
| Learning rate | 1e-3 | Fixed across all systems |
| Weight decay | 1e-4 | |
| Batch size | 256 | Fits the frozen-feature design in memory |
| Maximum epochs | 40 | Early stopping usually terminates far sooner |
| Early stopping | Patience 6 on validation macro-F1 | Validation split only |
| Gradient clipping | Norm 1.0 | |
| Class weighting | Inverse frequency on the training split | Guards the mild imbalance under Protocol B |
| Hidden and projection width | 256 | Shared by every fusion variant |
| Adversarial weight | 0.5, with the standard DANN ramp | Ganin *et al.* (2016) |
| Seeds | 13, 42, 7 | Python, NumPy and torch all seeded |
| Bootstrap replicates | 2,000 | Percentile method |
| Permutation replicates | 500 | Probe null, Section 6.2 |

## Appendix M: Plan against outcome

**Table M1:** Planned and actual delivery by phase.

| Phase | Planned content | Actual outcome | Variance and cause |
|:--|:----------------|:---------------|:-------------------|
| 1 | Literature review, research question, proposal, ethics | Delivered | Benchmark choice narrowed to Fakeddit earlier than planned |
| 2 | Corpus pipeline, baseline models | Corpus pipeline, three protocols, shortcut probe | Protocols brought forward; baselines deferred to Phase 3 after SC-01 |
| 3 | Fusion variants, evaluation | Model grid, ablations, adversarial variant, robustness suite | Two planned fusion variants dropped (SC-01); adversarial variant added |
| 4 | Writing | Rotation, prior decomposition, ranking analysis, triage output, writing | Additional analyses added after the internal audit; no milestone moved |

Milestone dates were met in every phase. The mechanism that made this possible
is described in Section 4.5: dates were fixed at phase boundaries while the
content of each iteration inside a phase remained open.

## Appendix N: Risk register

**Table N1:** Risks, mitigations and outcomes.

| # | Risk | Likelihood | Impact | Mitigation | Outcome |
|:--|:-----|:-----------|:-------|:-----------|:--------|
| R-01 | Reddit preview URLs expire, making the corpus unreproducible | High | High | Record fetch status for every row; release cached representations rather than requiring a rebuild | Materialised: 3.8% expired at build time. Absorbed |
| R-02 | Language-model feature extraction exceeds the available compute | High | Medium | Restrict the output head to one position per item | Materialised: 8 hours reduced to under 3. Absorbed |
| R-03 | A 165-run grid does not fit the laptop compute budget | Medium | High | Freeze encoders and cache representations | Materialised: grid runs in about 40 minutes. Absorbed |
| R-04 | The shortcut hypothesis is false and the project has no finding | Medium | High | Test it first with the cheapest possible probe | Did not materialise; the probe scored 1.000 |
| R-05 | The proposed architecture shows no advantage | Medium | Medium | Frame the contribution as measurement, so a null architectural result remains publishable | Materialised: reported honestly in Section 7.2 |
| R-06 | Personal data is inadvertently retained | Low | High | Drop the author column at build time; assert its absence in a unit test | Did not materialise |
| R-07 | A reported number drifts from the artefact output | Medium | High | Verify every figure against the result files before submission | Materialised: four rounding errors and one misstatement caught and corrected |

## Appendix O: Complete results

**Table O1:** All systems, Random protocol.

| System | Acc | macro-F1 (95% CI) | AUROC | AUPRC | Recall(false) | ECE | Brier |
|:-------|----:|:-----------------:|------:|------:|------:|------:|------:|
| Majority class | 0.519 | 0.342 (0.335-0.348) | 0.500 | 0.481 | 0.000 | 0.481 | 0.481 |
| Subreddit probe (no content) | 1.000 | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 0.005 | 0.000 |
| TF-IDF + LR | 0.752 | 0.751 (0.738-0.764) | 0.838 | 0.817 | 0.709 | 0.067 | 0.169 |
| LLM zero-shot judge | 0.511 | 0.453 (0.438-0.468) | 0.562 | 0.527 | 0.871 | 0.333 | 0.368 |
| LLM embedding + MLP | 0.781 | 0.781 (0.768-0.792) | 0.868 | 0.851 | 0.776 | 0.019 | 0.148 |
| Metadata only | 0.735 | 0.735 (0.721-0.747) | 0.810 | 0.795 | 0.746 | 0.030 | 0.178 |
| Consistency only | 0.679 | 0.669 (0.654-0.683) | 0.697 | 0.711 | 0.523 | 0.032 | 0.217 |
| Image only (CLIP) | 0.809 | 0.808 (0.796-0.820) | 0.890 | 0.894 | 0.790 | 0.023 | 0.133 |
| Text only (DistilRoBERTa) | 0.790 | 0.790 (0.778-0.802) | 0.875 | 0.865 | 0.799 | 0.052 | 0.146 |
| Late fusion | 0.842 | 0.842 (0.831-0.853) | 0.922 | 0.921 | 0.827 | 0.030 | 0.113 |
| Early fusion | 0.861 | 0.860 (0.850-0.871) | 0.934 | 0.932 | 0.847 | 0.038 | 0.103 |
| Early fusion + metadata | 0.900 | 0.900 (0.891-0.908) | 0.963 | 0.963 | 0.882 | 0.037 | 0.076 |
| Early fusion + adversarial | 0.851 | 0.851 (0.840-0.862) | 0.925 | 0.921 | 0.832 | 0.027 | 0.108 |
| CGF, no gate | 0.901 | 0.901 (0.892-0.910) | 0.963 | 0.961 | 0.884 | 0.037 | 0.075 |
| CGF, no consistency | 0.890 | 0.890 (0.880-0.898) | 0.956 | 0.956 | 0.863 | 0.036 | 0.083 |
| CGF, no interaction | 0.906 | 0.905 (0.897-0.914) | 0.966 | 0.965 | 0.901 | 0.038 | 0.073 |
| CGF, no metadata | 0.881 | 0.881 (0.872-0.891) | 0.950 | 0.948 | 0.859 | 0.043 | 0.089 |
| CGF | 0.905 | 0.905 (0.895-0.913) | 0.967 | 0.965 | 0.881 | 0.046 | 0.073 |
| CGF + adversarial | 0.892 | 0.892 (0.882-0.901) | 0.959 | 0.957 | 0.888 | 0.026 | 0.078 |
| CGF + LLM | 0.903 | 0.902 (0.894-0.911) | 0.964 | 0.962 | 0.901 | 0.041 | 0.074 |
| CGF + LLM + adversarial | 0.892 | 0.891 (0.882-0.901) | 0.957 | 0.954 | 0.866 | 0.041 | 0.082 |

**Table O2:** All systems, Source-disjoint protocol.

| System | Acc | macro-F1 (95% CI) | AUROC | AUPRC | Recall(false) | ECE | Brier |
|:-------|----:|:-----------------:|------:|------:|------:|------:|------:|
| Majority class | 0.405 | 0.288 (0.282-0.294) | 0.500 | 0.405 | 1.000 | 0.595 | 0.595 |
| Subreddit probe (no content) | 0.405 | 0.288 (0.282-0.294) | 0.500 | 0.405 | 1.000 | 0.249 | 0.303 |
| TF-IDF + LR | 0.626 | 0.616 (0.605-0.628) | 0.657 | 0.563 | 0.578 | 0.093 | 0.232 |
| LLM zero-shot judge | 0.423 | 0.386 (0.375-0.397) | 0.445 | 0.360 | 0.824 | 0.419 | 0.449 |
| LLM embedding + MLP | 0.570 | 0.563 (0.551-0.575) | 0.603 | 0.482 | 0.546 | 0.098 | 0.253 |
| Metadata only | 0.595 | 0.590 (0.578-0.602) | 0.614 | 0.488 | 0.598 | 0.110 | 0.252 |
| Consistency only | 0.610 | 0.562 (0.550-0.574) | 0.576 | 0.490 | 0.343 | 0.085 | 0.256 |
| Image only (CLIP) | 0.661 | 0.624 (0.612-0.636) | 0.700 | 0.611 | 0.430 | 0.153 | 0.238 |
| Text only (DistilRoBERTa) | 0.630 | 0.617 (0.605-0.628) | 0.668 | 0.519 | 0.552 | 0.125 | 0.244 |
| Late fusion | 0.669 | 0.647 (0.635-0.658) | 0.704 | 0.616 | 0.514 | 0.134 | 0.231 |
| Early fusion | 0.653 | 0.639 (0.627-0.650) | 0.697 | 0.584 | 0.566 | 0.158 | 0.245 |
| Early fusion + metadata | 0.686 | 0.649 (0.637-0.661) | 0.739 | 0.657 | 0.447 | 0.217 | 0.252 |
| Early fusion + adversarial | 0.649 | 0.636 (0.625-0.648) | 0.688 | 0.592 | 0.567 | 0.169 | 0.251 |
| CGF, no gate | 0.693 | 0.665 (0.653-0.677) | 0.754 | 0.644 | 0.501 | 0.179 | 0.234 |
| CGF, no consistency | 0.694 | 0.679 (0.668-0.690) | 0.745 | 0.664 | 0.586 | 0.200 | 0.244 |
| CGF, no interaction | 0.678 | 0.641 (0.629-0.652) | 0.714 | 0.629 | 0.436 | 0.213 | 0.258 |
| CGF, no metadata | 0.644 | 0.606 (0.593-0.617) | 0.685 | 0.562 | 0.411 | 0.228 | 0.276 |
| CGF | 0.668 | 0.625 (0.612-0.636) | 0.715 | 0.621 | 0.406 | 0.223 | 0.264 |
| CGF + adversarial | 0.687 | 0.653 (0.641-0.664) | 0.734 | 0.637 | 0.458 | 0.162 | 0.231 |
| CGF + LLM | 0.680 | 0.642 (0.629-0.654) | 0.739 | 0.628 | 0.433 | 0.179 | 0.237 |
| CGF + LLM + adversarial | 0.689 | 0.672 (0.661-0.684) | 0.744 | 0.630 | 0.572 | 0.138 | 0.222 |

**Table O3:** All systems, Temporal protocol.

| System | Acc | macro-F1 (95% CI) | AUROC | AUPRC | Recall(false) | ECE | Brier |
|:-------|----:|:-----------------:|------:|------:|------:|------:|------:|
| Majority class | 0.524 | 0.344 (0.337-0.350) | 0.500 | 0.476 | 0.000 | 0.476 | 0.476 |
| Subreddit probe (no content) | 0.976 | 0.976 (0.972-0.981) | 0.988 | 0.983 | 0.950 | 0.116 | 0.067 |
| TF-IDF + LR | 0.617 | 0.616 (0.602-0.630) | 0.659 | 0.594 | 0.701 | 0.110 | 0.243 |
| LLM zero-shot judge | 0.523 | 0.471 (0.457-0.486) | 0.579 | 0.532 | 0.878 | 0.317 | 0.358 |
| LLM embedding + MLP | 0.659 | 0.650 (0.635-0.663) | 0.675 | 0.578 | 0.867 | 0.165 | 0.254 |
| Metadata only | 0.695 | 0.694 (0.681-0.708) | 0.772 | 0.737 | 0.782 | 0.056 | 0.197 |
| Consistency only | 0.697 | 0.686 (0.672-0.700) | 0.739 | 0.750 | 0.540 | 0.035 | 0.211 |
| Image only (CLIP) | 0.647 | 0.641 (0.626-0.655) | 0.716 | 0.697 | 0.822 | 0.213 | 0.276 |
| Text only (DistilRoBERTa) | 0.649 | 0.639 (0.625-0.653) | 0.672 | 0.580 | 0.850 | 0.179 | 0.262 |
| Late fusion | 0.667 | 0.662 (0.648-0.675) | 0.726 | 0.690 | 0.832 | 0.216 | 0.267 |
| Early fusion | 0.687 | 0.678 (0.664-0.692) | 0.763 | 0.726 | 0.892 | 0.237 | 0.269 |
| Early fusion + metadata | 0.725 | 0.720 (0.707-0.733) | 0.842 | 0.829 | 0.896 | 0.211 | 0.232 |
| Early fusion + adversarial | 0.684 | 0.677 (0.662-0.690) | 0.742 | 0.666 | 0.882 | 0.211 | 0.260 |
| CGF, no gate | 0.740 | 0.738 (0.725-0.751) | 0.842 | 0.829 | 0.872 | 0.195 | 0.221 |
| CGF, no consistency | 0.706 | 0.702 (0.687-0.714) | 0.798 | 0.780 | 0.877 | 0.226 | 0.253 |
| CGF, no interaction | 0.748 | 0.745 (0.732-0.757) | 0.862 | 0.851 | 0.906 | 0.186 | 0.209 |
| CGF, no metadata | 0.703 | 0.696 (0.682-0.708) | 0.791 | 0.762 | 0.896 | 0.229 | 0.253 |
| CGF | 0.746 | 0.743 (0.730-0.754) | 0.850 | 0.834 | 0.893 | 0.197 | 0.217 |
| CGF + adversarial | 0.733 | 0.730 (0.717-0.742) | 0.826 | 0.791 | 0.880 | 0.199 | 0.225 |
| CGF + LLM | 0.726 | 0.721 (0.708-0.734) | 0.839 | 0.828 | 0.900 | 0.212 | 0.231 |
| CGF + LLM + adversarial | 0.725 | 0.721 (0.707-0.734) | 0.812 | 0.773 | 0.902 | 0.206 | 0.231 |

**Table O4:** Response to synthetic de-contextualisation, source-disjoint protocol, genuine posts only.

| System | P(false) clean | P(false) shuffled | Change | Flipped to false |
|:-------|----:|----:|----:|----:|
| CGF | 0.203 | 0.434 | +0.231 | 32.8% |
| CGF + LLM | 0.235 | 0.443 | +0.208 | 32.1% |
| Early fusion | 0.314 | 0.491 | +0.177 | 28.4% |
| Image only (CLIP) | 0.253 | 0.434 | +0.181 | 32.4% |
| Text only (DistilRoBERTa) | 0.364 | 0.364 | +0.000 | 0.0% |

**Table O5:** Test-time robustness, macro-F1 by condition, source-disjoint protocol.

| System | clean | text_lowercase | text_no_punct | text_typos | text_word_dropout | zero_cons | zero_image | zero_meta | zero_text |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CGF | 0.632 | 0.630 | 0.630 | 0.622 | 0.641 | 0.635 | 0.569 | 0.606 | 0.638 |
| CGF + LLM | 0.650 | - | - | - | - | 0.656 | 0.601 | 0.631 | 0.626 |
| Early fusion | 0.639 | - | - | - | - | 0.639 | 0.498 | 0.639 | 0.623 |
| Image only (CLIP) | 0.618 | - | - | - | - | 0.618 | 0.288 | 0.618 | 0.618 |
| Text only (DistilRoBERTa) | 0.618 | - | - | - | - | 0.618 | 0.618 | 0.618 | 0.288 |

**Table O6:** Ranking quality on the source-disjoint test set. Prevalence of false items is 0.405.

| System | P@50 | P@100 | P@500 | P@1000 | Lift@100 |
|:---|---:|---:|---:|---:|---:|
| Early fusion + metadata | 0.880 | 0.860 | 0.804 | 0.744 | 2.12 |
| Image only (CLIP) | 0.840 | 0.850 | 0.770 | 0.696 | 2.10 |
| CGF | 0.820 | 0.820 | 0.754 | 0.718 | 2.02 |
| CGF + adversarial | 0.780 | 0.780 | 0.742 | 0.729 | 1.92 |
| CGF + LLM + adversarial | 0.740 | 0.750 | 0.704 | 0.724 | 1.85 |
| TF-IDF + LR | 0.700 | 0.720 | 0.674 | 0.655 | 1.78 |
| Subreddit probe (no content) | 0.440 | 0.400 | 0.392 | 0.400 | 0.99 |
| Text only (DistilRoBERTa) | 0.440 | 0.400 | 0.508 | 0.548 | 0.99 |
| LLM zero-shot judge | 0.260 | 0.250 | 0.270 | 0.295 | 0.62 |

**Table O7:** Leave-one-community-out rotation, per fold, accuracy at a 0.5 threshold.

| Held-out community | n | Label | Text only | Early fusion + meta | CGF | CGF + adv |
|:---|---:|:---|---:|---:|---:|---:|
| r/mildlyinteresting | 1380 | true | 0.083 | 0.109 | 0.193 | 0.146 |
| r/pic | 1128 | true | 0.062 | 0.176 | 0.306 | 0.245 |
| r/photoshopbattles | 1535 | true | 0.056 | 0.179 | 0.341 | 0.334 |
| r/savedyouaclick | 1598 | fake | 0.415 | 0.431 | 0.456 | 0.460 |
| r/misleadingthumbnails | 1524 | fake | 0.222 | 0.372 | 0.457 | 0.469 |
| r/pareidolia | 1521 | fake | 0.657 | 0.480 | 0.488 | 0.218 |
| r/fakehistoryporn | 1438 | fake | 0.267 | 0.556 | 0.595 | 0.644 |
| r/confusing_perspective | 1187 | fake | 0.826 | 0.788 | 0.612 | 0.684 |
| r/propagandaposters | 1561 | fake | 0.771 | 0.912 | 0.644 | 0.792 |
| r/subredditsimulator | 1555 | fake | 0.550 | 0.293 | 0.645 | 0.582 |
| r/nottheonion | 1596 | true | 0.674 | 0.776 | 0.682 | 0.717 |
| r/upliftingnews | 1598 | true | 0.802 | 0.839 | 0.870 | 0.864 |
| r/usnews | 1261 | true | 0.752 | 0.904 | 0.948 | 0.923 |
| r/neutralnews | 1579 | true | 0.807 | 0.939 | 0.948 | 0.966 |
| r/usanews | 1228 | true | 0.892 | 0.971 | 0.979 | 0.976 |

**Table O8:** Per-held-out-community performance of the proposed system, source-disjoint protocol.

| Held-out community | n | Label | Accuracy | Mean P(false) |
|:---|---:|:---|---:|---:|
| r/confusing_perspective | 1187 | fake | 0.472 | 0.476 |
| r/savedyouaclick | 1598 | fake | 0.646 | 0.605 |
| r/nottheonion | 1596 | true | 0.667 | 0.361 |
| r/usanews | 1228 | true | 0.827 | 0.211 |
| r/usnews | 1261 | true | 0.841 | 0.211 |

**Table O9:** Pairwise comparisons under the source-disjoint protocol, with Holm-Bonferroni correction across the family of twenty tests.

| System vs CGF+LLM+adversarial | delta macro-F1 (95% CI) | p (raw) | p (Holm) | Significant |
|:---|:---|---:|---:|:---:|
| Subreddit probe (no content) | +0.384 (+0.371 to +0.397) | <0.001 | <0.001 | yes |
| Majority class | +0.384 (+0.371 to +0.397) | <0.001 | <0.001 | yes |
| LLM zero-shot judge | +0.287 (+0.271 to +0.303) | <0.001 | <0.001 | yes |
| LLM embedding + MLP | +0.109 (+0.095 to +0.124) | <0.001 | <0.001 | yes |
| Metadata only | +0.082 (+0.067 to +0.097) | <0.001 | <0.001 | yes |
| Consistency only | +0.110 (+0.096 to +0.125) | <0.001 | <0.001 | yes |
| CGF, no metadata | +0.067 (+0.058 to +0.077) | <0.001 | <0.001 | yes |
| Text only (DistilRoBERTa) | +0.055 (+0.042 to +0.069) | <0.001 | <0.001 | yes |
| TF-IDF + LR | +0.056 (+0.041 to +0.071) | <0.001 | <0.001 | yes |
| Early fusion + adversarial | +0.036 (+0.026 to +0.047) | <0.001 | <0.001 | yes |
| Early fusion | +0.033 (+0.023 to +0.045) | <0.001 | <0.001 | yes |
| Image only (CLIP) | +0.048 (+0.036 to +0.060) | <0.001 | <0.001 | yes |
| CGF | +0.048 (+0.038 to +0.057) | <0.001 | <0.001 | yes |
| Late fusion | +0.026 (+0.015 to +0.036) | <0.001 | <0.001 | yes |
| CGF, no interaction | +0.032 (+0.022 to +0.041) | 0.022 | 0.133 | no |
| CGF + LLM | +0.031 (+0.022 to +0.040) | 0.049 | 0.244 | no |
| CGF, no consistency | -0.007 (-0.017 to +0.004) | 0.282 | 1.000 | no |
| CGF, no gate | +0.007 (+0.000 to +0.015) | 0.293 | 1.000 | no |
| Early fusion + metadata | +0.023 (+0.011 to +0.035) | 0.603 | 1.000 | no |
| CGF + adversarial | +0.020 (+0.011 to +0.028) | 0.732 | 1.000 | no |

## Appendix P: The reviewer triage queue

The artefact emits the queue itself rather than only the metric that scores it.
Each row carries the calibrated probability, the empirical proportion of items
in the same score band that were in fact false, and the two per-item signals a
reviewer can check against the post. The head of the queue is dominated by
clickbait constructions, which is the qualitative counterpart of the precision
figure in Section 6.9.

**Table P1:** Head of the triage queue, proposed system, source-disjoint protocol.

| Rank | P(false) | Band precision | CLIP consistency | Gate | Headline |
|---:|---:|---:|---:|---:|:---|
| 1 | 1.000 | 0.712 | 0.166 | 0.474 | robert downey jr explains why he and chris evans left the marvel cinemat |
| 2 | 0.999 | 0.712 | 0.167 | 0.475 | a yearold man paraded his yearold bride in times square to bring home a  |
| 3 | 0.999 | 0.712 | 0.151 | 0.479 | half life update fans hopeful after valve make this shock announcement |
| 4 | 0.999 | 0.712 | 0.151 | 0.483 | white house puts out fabricated video and transcript of helsinki press c |
| 5 | 0.999 | 0.712 | 0.116 | 0.475 | dog the bounty hunter receives scary diagnosis months after wifes death |
| 6 | 0.999 | 0.712 | 0.207 | 0.492 | glowing nippy |
| 7 | 0.999 | 0.712 | 0.140 | 0.470 | tears alert this soldier came back from the war and what he saw on his r |
| 8 | 0.999 | 0.712 | 0.248 | 0.497 | nigeria is ebola meme being used to spread fears virus was created by wh |
| 9 | 0.999 | 0.712 | 0.100 | 0.479 | school dominated by black and asian pupils will lead to a sexual volcano |
| 10 | 0.999 | 0.712 | 0.154 | 0.476 | new heinz condiment mayochup has an unfortunate translation in cree |
| 11 | 0.999 | 0.712 | 0.144 | 0.482 | you will never guess why this man built this pool |
| 12 | 0.999 | 0.712 | 0.207 | 0.483 | ibms watson memorized the entire urban dictionary then his overlords had |

Of the top 100 items, 75 are in fact false against a prevalence of 40.5%.

## Appendix Q: Extended ethical analysis

Section 5.4 states the four decisions that changed the artefact. This appendix
records the reasoning in full.

**Legal basis and minimisation.** The corpus contains pseudonymous identifiers,
which are personal data under UK GDPR. Processing rests on legitimate
interests, read with the research provisions, and the balancing test turns on
necessity: because no model requires the author field, retaining it could not be
necessary, so it is dropped at build time and a unit test asserts its absence.

**Contextual integrity.** Nissenbaum's (2010) framework asks whether a flow of
information violates the norms of the context in which it was shared. Posting
publicly to a subreddit anticipates being read and quoted within that community
and by researchers studying it in aggregate; it does not anticipate being
reproduced beside a machine judgement of truthfulness. The report therefore
reports only aggregates and reproduces no individual post, and the triage
extract in Appendix P shows headlines drawn from communities that are labelled
by construction rather than by any judgement about an individual author.

**Public availability is not consent.** franzke *et al.* (2020) argue that
accessibility does not settle the ethical question, and Fiesler and Proferes
(2018) show empirically that users' expectations about research use diverge
sharply from platform terms. The response adopted here is to treat minimisation
as a technical control rather than a promise, and to treat an expired preview
URL as a signal that content may have been withdrawn, in which case no attempt
is made to recover it from another source.

**Dual use.** Publishing a documented shortcut and an adversarial remedy also
documents what an adversary must defeat. Weighing this, a confound that inflates
every published figure on a widely used benchmark is more dangerous unstated
than stated, since it currently misleads every reader of that literature,
whereas the evasion value of the disclosure is small: the shortcut is a property
of the benchmark, not of any deployed system. The release therefore carries a
use restriction prohibiting automated removal, together with a model card
(Mitchell *et al.*, 2019) recording intended and prohibited use.

**Disaggregation as an ethical control.** A single aggregate accuracy would
conceal that the system is correct on 84.1% of posts from one held-out community
and 47.2% from another (Appendix O, Table O8). Reporting the disaggregation is
what makes the deployment recommendation in Section 5.4 evidence-based rather
than precautionary.

**Environmental cost.** The pipeline requires approximately five CPU-hours and
no GPU. Strubell, Ganesh and McCallum (2019) report figures four orders of
magnitude larger for transformer training, which is part of the argument for
freezing the encoders rather than fine-tuning every variant.

## Appendix R: Comparison of prior multimodal detectors

**Table R1:** Representative multimodal detectors: corpus, fusion mechanism, evaluation split and the limitation each carries for this study. Reported accuracies range from 0.606 to 0.906, all but one obtained under a random split.

**Table 1:** Representative multimodal detectors: corpus, fusion, evaluation split and the limitation each carries for this study.

| Work | Corpus | Fusion | Split | Limitation for this study |
|:-----|:-------|:-------|:------|:--------------------------|
| Jin *et al.* (2017) | Weibo, Twitter | Attention RNN | Random | Social features unavailable at posting time |
| Wang *et al.* (2018) | Weibo, Twitter | Event-adversarial | Event-disjoint | Invariance to event, not to source |
| Khattar *et al.* (2019) | Weibo, Twitter | Shared latent space | Random | Latent space is not inspectable |
| Zhou, Wu and Zafarani (2020) | PolitiFact, GossipCop | Explicit similarity | Random | Similarity treated as the prediction |
| Nakamura *et al.* (2020) | Fakeddit | Late fusion | Random | Labels are subreddits; the split shares sources |
| Luo *et al.* (2021) | NewsCLIPpings | CLIP | Random | Mismatch is synthetic, not naturally labelled |
| Alam *et al.* (2022) | Survey, 30+ systems | Various | Predominantly random | Records no source-disjoint protocol |
| Shen *et al.* (2024) | Weibo, Twitter | Contrastive, optimal transport | Random | No source-shift condition |
| Qi *et al.* (2024) | NewsCLIPpings | Vision-language model | Random | Inference cost not reported |

## Appendix S: Objectives, success criteria and outcomes

**Table S1:** Objectives with measurable success criteria, schedule and outcome.

| # | Objective | Success criterion | By | Ch. | Outcome |
|:--|:----------|:------------------|:--:|:--:|:--------|
| O1 | Critically review misinformation detection, multimodal fusion and evaluation methodology | 30 or more sources; prior split protocols tabulated; a methodological gap identified | Dec | 2 | Partially met: 32 sources, but 30 reached only after Phase 4 |
| O2 | Build a reproducible multimodal corpus and three evaluation protocols | 20,000 or more posts; automated build; split disjointness verified by test | Jan | 4 | Met: 21,798 posts; disjointness asserted in the test suite |
| O3 | Quantify the source-leakage shortcut | Content-free probe with a bootstrap interval and a permutation null | Feb | 6 | Met: 1.000 macro-F1, null 0.408, *p* < 0.01 |
| O4 | Design a consistency-aware detector and integrate a language model | Working artefact with a component ablation | Mar | 5, 6 | Met as an artefact; the performance claim is withdrawn in Section 7.2 |
| O5 | Evaluate against 10 or more baselines with quantified uncertainty | Three seeds; bootstrap intervals; corrected significance tests | Apr | 6 | Met: 21 systems, 165 runs, Holm-corrected tests |
| O6 | Test robustness, ranking quality and behaviour beyond accuracy | Modality ablation, out-of-context probe, calibration, precision at k, measured cost | May | 6 | Met: all five reported |
| O7 | Implement three ethical constraints affecting the artefact | Minimisation, threshold-free output and disaggregated reporting, each traceable to a module and a reported metric | Jun | 5, 6 | Met: `build_dataset.py`, F2 and ECE, Table O8 |

## Appendix T: Process and design diagrams

The five diagrams below, together with Figure 1, are the project's
flowcharts. Figure 1 combines the pipeline and the model on a single page;
`diagrams/architecture.drawio` is that combined page, and `model.drawio` and
`pipeline.drawio` hold the two panels separately for reuse.
Each is supplied as an editable draw.io file in `diagrams/`, and each figure in
this report is rendered from the same declaration that produces the editable
file, so the two cannot drift apart.

Colour carries meaning rather than decoration. A warm sand terminator opens and
closes a flow; soft blue is an ordinary process step; white is a stored
artefact; gold is a decision point; deeper amber marks an element this project
contributes; and muted red marks an adversarial branch or a corrective taken
against the project's own claim. Every stroke is darker than its fill, so the
shapes remain distinguishable in greyscale printing.

**Table T1:** Diagram files.

| Figure | Diagram | Editable source |
|:-------|:--------|:----------------|
| Figure 1 | Artefact architecture, both panels on one page | `diagrams/architecture.drawio` |
| Figure 1(b) | Consistency-Gated Fusion alone | `diagrams/model.drawio` |
| Figure T1 | Research process | `diagrams/research.drawio` |
| Figure T2 | Processing pipeline, vertical form | `diagrams/pipeline.drawio` |
| Figure T3 | Evaluation protocol design | `diagrams/protocols.drawio` |
| Figure T4 | Ethics to artefact traceability | `diagrams/ethics.drawio` |
| Figure T5 | Development cycles and scope decisions | `diagrams/cycles.drawio` |
| all | Every page in one workbook | `diagrams/mmid-diagrams.drawio` |

![**Figure T1.** The research process actually followed, including the corrective branch taken when the shortcut probe returned a perfect score.](../figures/flow_research.png)

![**Figure T2.** The six processing stages and the cached artefact each one writes.](../figures/flow_pipeline.png)

![**Figure T3.** How one corpus is split three ways, and the three diagnostics applied to the result.](../figures/flow_protocols.png)

![**Figure T4.** Each ethical position, the decision it forced in the artefact, and where a marker can verify that it did.](../figures/flow_ethics.png)

![**Figure T5.** The eight development cycles and the two scope decisions their results forced.](../figures/flow_cycles.png)
