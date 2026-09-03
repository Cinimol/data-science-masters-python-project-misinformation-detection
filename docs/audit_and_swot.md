---
title: "Rubric Audit and SWOT Analysis"
subtitle: "Multimodal Detection of Misinformation Using Deep Learning and Large Language Models, UFCF9Y-60-M"
---

# 1. Method

The dissertation was audited against the eight marking criteria in the *Artefact and Report Assessment Specification* (25SEP) and against the mandatory items in *Project report structure*. Every criterion was scored against its published 70+ descriptor, not against a general impression. In parallel, every numeric claim in the report was recomputed from the artefact's own result files, and every bibliography entry was checked against its publisher record.

The audit was deliberately adversarial: the brief was to find reasons a second marker could withhold marks, not to confirm that the work was good. What follows is the result, the fixes applied, and what remains outside anyone's control but the student's.

# 2. Audit result

**Table 1:** Criterion-by-criterion assessment, before and after remediation.

| Criterion | Marks | Before | After | Decisive evidence for the revised position |
|:----------|------:|-------:|------:|:-------------------------------------------|
| Rationale, scope and objectives | 5 | 3.5 | 4.5 | Objectives now carry measurable success criteria, scheduled dates and a reported outcome against each (Appendix S); scope explicitly bounds the language-model claim |
| Novelty and innovation | 10 | 7 | 9 | Content-free probe as a benchmark diagnostic; leave-one-community-out rotation; prior-shift decomposition; reviewer triage output; figures implemented twice |
| Literature review and academic content | 20 | 13.5 | 17.5 | 44 verified sources against 25; comparison of nine prior systems by evaluation protocol (Appendix R); contribution positioned against the two nearest critiques |
| Design and approach, user requirements | 15 | 10 | 13 | Documented evidence base for every user need (Appendix J); requirements with acceptance criteria (Appendix G); hyperparameters fixed and justified (Appendix L); two design decisions tested rather than asserted |
| Implementation | 15 | 11 | 13.5 | 25 automated tests including a data-minimisation assertion; triage queue as a working output; architecture figure generated from code; every reported number machine-verified |
| Ethical consideration and approach | 10 | 8 | 9 | Legal basis, contextual integrity, dual-use judgement, model card and disaggregated reporting, each traced to a code decision or a reported metric (Section 5.4, Appendix Q) |
| Evaluation and reflection, planning and management | 15 | 8 | 12 | Rotation, prior decomposition, corrected significance testing, per-community reporting, an explicit evaluation of the approach (Section 7.3), iteration log, risk register and plan-versus-outcome. **Capped until Appendices A and B are completed** |
| Presentation, organisation, documentation, attribution | 10 | 6 | 8.5 | All 39 defects closed; every figure and table cited in text; complete referencing; consistent counts across report, README and code |
| **Total** | **100** | **67** | **87** | Subject to the three student-only items in Section 5 |

The single largest movement is in *Evaluation and reflection*, and it is also the most fragile: roughly three of those marks depend on records only the student holds.

# 3. What was found, and what was done

## 3.1 Substantive weaknesses

**The central claim had an untested alternative explanation.** Holding out whole communities changes the class balance of the test set as well as its sources, so a sceptical marker could have attributed the 0.213 macro-F1 protocol gap to prior shift rather than to leakage. The source-disjoint test set was resampled to the random split's class prior and every system re-scored: the gap is 0.205 before matching and 0.209 after, so prior shift explains none of it. The project's headline finding now survives the obvious attack on it.

**The architectural claim was not identifiable.** Protocol B used one fixed partition of five communities, so its numbers carried a selection uncertainty that item-level bootstrap intervals cannot capture. A full leave-one-community-out rotation over fifteen communities was run. The standard deviation across folds is 0.245 accuracy against 0.029 separating the best and worst architectures, and the proposed system is correct on fewer than half the posts in six of fifteen folds. The report now states plainly that the architectural comparison is not identifiable at this corpus size, and Section 7.2 withdraws the claim. This is a stronger position than the one it replaces: a report arguing that the field over-claims on inadequate evaluation cannot exempt itself.

**The stated requirement and the reported metric did not match.** Chapter 2 required a ranked review queue; Chapter 6 reported only macro-F1. Precision at k is now reported, and it orders the systems differently, with a simpler baseline achieving the highest precision at 100 items. The artefact also emits the queue itself, with per-row calibrated probability, empirical band precision and the two inspectable signals.

**Significance was uncorrected across sixty comparisons.** Holm-Bonferroni correction is now applied within protocol; fourteen of twenty comparisons survive under the source-disjoint protocol.

**The headline result had no uncertainty.** The content-free probe is now reported with a bootstrap interval and a permutation null of 0.408 (SD 0.097, maximum 0.705 over 500 permutations).

**An ethical commitment was made and not discharged.** Section 5.4 promised disaggregated reporting; nothing was disaggregated. Per-community results are now reported, and they matter: accuracy ranges from 0.472 to 0.841 across the five held-out communities, a spread sixteen times the margin separating the leading systems.

## 3.2 Structural and documentary weaknesses

**"Evaluation of general approach" is mandatory and was absent.** The material existed but was scattered across three sections and never signposted. Section 7.3 now evaluates the approach as a research instrument, including the finding that the instrument is coarser than the question.

**User needs rested on assertion.** Every requirement is now traced in Appendix J to a named source and the specific extract relied on, drawn from regulation, moderation practice, human-AI decision studies and enforcement reporting.

**Evidence of planning and management was absent.** Section 4.5, plus an iteration log with scope decisions (Appendix K), a plan-against-outcome comparison (Appendix M) and a seven-entry risk register recording which risks materialised and how they were absorbed (Appendix N).

**Hyperparameters were undocumented, which undermined the comparability argument.** Appendix L records the shared configuration and states that no per-model search was performed, so no reported comparison is contaminated by selection over hyperparameters.

## 3.3 Accuracy and presentation

Thirty-nine defects were identified and closed. The material ones were factual: the report named the wrong system as best under the corrected protocol (a CGF ablation at 0.679 outperforms the proposed system at 0.672, and both are now shown); "less than two thirds survives" was unsupported by any stated figure; the run-count arithmetic did not work; the language-model judge was called "calibrated" in Chapter 5 and shown to be badly miscalibrated in Chapter 6; and nine metadata features were enumerated as eight. Five of eight figures and all eleven tables were never referred to in the text. Access dates were absent from every reference, and one reference carried a fabricated author list.

The bibliography was re-verified entry by entry against publisher records. It now holds 44 sources, every one of which resolves and every one of which is cited.

# 4. SWOT analysis

## Strengths

The contribution is methodological rather than incremental, which is the rarer and more defensible kind. A content-free probe scoring 1.000 is a result a marker can verify in one line of the results file, and it reframes an entire literature rather than adding to it.

Two central design decisions are tested rather than asserted. Freezing the encoders was checked against an end-to-end fine-tuned control and cost no accuracy under source shift; the protocol gap was checked against prior shift and survived. Work that tries to break its own premises reads differently from work that only supports them.

The honesty is now structural rather than rhetorical. The report withdraws its own architectural claim on the evidence, reports a simpler baseline beating the proposed system on the operative metric, and records that an objective was missed. Against a rubric that rewards "deep and honest reflection", this is worth more than a defended claim would have been.

Reproducibility is unusually complete: 165 seeded runs, a fifteen-fold rotation, 25 tests, every figure implemented twice, every number machine-verified against the artefact's own outputs, and the whole pipeline runs on a laptop CPU in about five hours.

## Weaknesses

A single benchmark cannot establish that the confound is general. The contribution is a demonstrated mechanism plus an argument by analogy to the dataset-bias literature, and the report says so.

The proposed architecture does not win. Under the corrected protocol two of its own ablations are ahead of it, and a simpler baseline ranks better on precision at 100. This is reported rather than concealed, but it remains a weakness in the artefact even as it strengthens the argument.

User needs are documented rather than validated. No practitioner was interviewed, and no amount of careful sourcing fully substitutes for that.

The rotation uses one seed per fold, so a small part of the reported fold-to-fold spread is seed variance.

## Opportunities

Three moves would raise the mark further and are available before submission. Completing Appendices A and B is worth roughly three marks on a fifteen-mark criterion and requires records the student already holds. Refreshing the Word table of contents takes one keystroke and prevents a marker opening the file to an empty field. Publishing the repository and pasting the URL into the Blackboard comments box is a stated deliverable.

Beyond submission, the leave-one-community-out finding is publishable in its own right. A short paper reporting that architectural comparison is not identifiable on a distantly supervised corpus at this scale would stand independently of the detector.

## Threats

The three student-only items are the principal threat. A blank supervision record and a missing ethics form are not presentational faults; they are mandatory items, and their absence caps a fifteen-mark criterion regardless of how strong the evaluation is.

A marker who reads only Chapter 6 could mistake the withdrawal of the architectural claim for a failed project. Section 7.2 states the position explicitly for that reason, but the risk is real, and the viva is the place to make the argument that a correctly measured null result is the finding.

The word limit is binding at 5,993 words against a 6,000 ceiling. Any further addition requires a corresponding deletion.

Finally, the artefact depends on Reddit preview URLs that continue to expire. The cached representations shipped with the release are what make the reported numbers reproducible, and the release notes say so.

# 5. What only the student can complete

**Table 2:** Outstanding items, all of which are mandatory or stated deliverables.

| # | Item | Where | Effect if omitted |
|:--|:-----|:------|:------------------|
| 1 | Supervision meeting record: dates, discussion, decisions | Appendix A, Table A1 | Mandatory item. Caps the 15-mark evaluation and management criterion |
| 2 | Signed ethical approval form from the supervisor | Appendix B | Mandatory item. Section 5.4 cites it as evidence |
| 3 | Declaration of generative AI use | Appendix I | University submission requirement |
| 4 | Student number and name on the title page; rename the file to `<student number> report.docx` | Front matter and filename | Stated submission convention |
| 5 | Refresh the table of contents in Word (select all, then F9) | Page 2 | The field is present but unpopulated until refreshed |
| 6 | Publish the repository and paste the URL into the Blackboard comments box, and into Appendix C | Appendix C | Stated deliverable in Section 3 of the assessment specification |

Items 1 to 3 cannot be supplied by anyone else, and fabricating them would be an assessment offence. Items 4 to 6 take a few minutes each.

# 6. Position

With the six items above completed, the submission sits in the low-to-mid 80s on a considered reading of the criteria, and the argument for the upper band rests on three things a marker can check quickly: a headline result that is verifiable in one line of the artefact's output, a project that tests its own premises and reports what happens when they fail, and an evaluation that is honest about the limits of its own instrument.

Without items 1 and 2 it sits several marks lower, for reasons that have nothing to do with the quality of the work.
