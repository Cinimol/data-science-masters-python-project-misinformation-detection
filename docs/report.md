---
title: "Multimodal Detection of Misinformation Using Deep Learning and Large Language Models"
subtitle: "Source leakage in a multimodal fake-news benchmark, and what an honest protocol shows"
author: "MSc Data Science, UFCF9Y-60-M CSCT Masters Project, UWE Bristol"
---

# Abstract 

Automated misinformation detection is routinely reported above 90% accuracy on public multimodal benchmarks, yet deployed systems generalize poorly. Using Fakeddit, this project asks whether those figures measure veracity detection or a more readily learnable property of the data. A probe classifier that sees only the name of the source community, with no text, image or metadata, reaches 100.0% accuracy under the conventional random split against a permutation null of 0.408, establishing source identity as a complete shortcut. Under a source-disjoint protocol, the average system loses 0.213 macro-F1, and resampling the test set to the conventional class balance shows that class-prior shift explains none of that loss. A leave-one-community-out rotation moves accuracy by a standard deviation of 0.245, an order of magnitude above the 0.029 separating the best and worst architectures tested, so architectural comparison on a single held-out partition is not identifiable at this corpus size. The work contributes a quantified benchmark confound, a corrected evaluation protocol honest about its own limits, and an open artefact that reproduces every result on a laptop CPU

# Chapter 1: Introduction

## 1.1 Context of the project

Misinformation, defined here as false or misleading claims circulated without regard for accuracy, is a structural feature of online information environments (Zhou and Zafarani, 2020). It is now overwhelmingly *multimodal*: a claim arrives as a headline attached to a photograph that is frequently authentic but taken from an unrelated event (Alam *et al.*, 2022). Generative models also produce fluent text and photorealistic imagery at negligible cost, removing the surface cues earlier detectors relied on (Chen and Shu, 2024). Regulation has followed: Articles 34 and 35 of the Digital Services Act oblige very large platforms to mitigate systemic risks from disinformation, and Article 20(6) requires that moderation complaints not be resolved by automated means alone (European Union, 2022).

## 1.2 Rationale

The literature suggests the problem is close to solved: the detectors surveyed in Appendix R report accuracies from 0.606 to 0.906. Practitioner experience does not corroborate this: models trained on one corpus degrade sharply on another and platforms still rely heavily on human review (Roberts, 2019). The discrepancy admits an explanation the field has been slow to confront. Large benchmarks cannot be hand-labelled at scale, so labels come from *distant supervision*, inferred from a document's source. Fakeddit (Nakamura, Levy and Wang, 2020), the largest public multimodal benchmark, labels a Reddit submission by the subreddit it was posted in: every item from r/nottheonion is treated as true and every item from r/propagandaposters as false. Where the split is random, source communities appear on both sides of it, and a model can score highly by recognising community style, such as vocabulary, image aesthetics and posting conventions, while learning nothing about whether claims are true. This is a textbook instance of shortcut learning (Geirhos *et al.*, 2020) and would inflate every result relying on a random split, which this project tests directly rather than assumes.

## 1.3 Problem statement and scope

**Problem.** Reported performance of multimodal misinformation detectors may be substantially attributable to source leakage in benchmark construction rather than to veracity modelling, and the field lacks a protocol separating the two.

**In scope.** Binary classification of English-language posts comprising a headline, one image and platform metadata; the Fakeddit benchmark; frozen encoders with trainable fusion; evaluation under three protocols with uncertainty quantification and significance testing. "Language model" denotes a 0.5B-parameter open-weight instruction-tuned model chosen for laptop reproducibility; larger and vision-language models are out of scope, and the negative result of Section 6.7 is not generalised to them.

**Out of scope.** Evidence retrieval and claim verification, surveyed by Thorne and Vlachos (2018); video and audio; propagation signals, unavailable at posting time; non-English content; and automated removal, for the reasons in Section 5.4.

## 1.4 Aims and objectives

**Aim.** To determine how much of the reported performance of multimodal misinformation detectors survives an evaluation protocol that removes source leakage, and to design and evaluate a detector that improves on baselines under that stricter protocol.

Seven objectives follow: O1 review detection, fusion and evaluation methodology; O2 build a reproducible corpus and three protocols; O3 quantify the source-leakage shortcut; O4 design a consistency-aware detector with a language model; O5 evaluate against ten or more baselines with quantified uncertainty; O6 test robustness, ranking and behaviour beyond accuracy; O7 implement three ethical constraints that change the artefact. Appendix S gives each with its measurable success criterion, scheduled date and discharging chapter; Section 7.2 reports the outcome against every criterion.

## 1.5 Structure of this report

Chapter 2 reviews the literature, derives user needs and locates the gap; Chapter 3 specifies the artefact; Chapter 4 covers the corpus, protocols, methodology and project management; Chapter 5 implementation and ethics; Chapter 6 the evaluation; Chapter 7 the discussion, evaluation of the approach and reflection; and Chapter 8 concludes.

# Chapter 2: Literature Review

## 2.1 From feature engineering to representation learning

Shu *et al.* (2017) establish the baselines still used today, while Rashkin *et al.* (2017) show that stylistic markers such as hedging and subjectivity separate reliable from unreliable news. The implication is ambiguous, since a model that learns style acquires genuine information about how a community writes but not about whether a claim is true. The decisive shift came with pre-trained transformers, notably BERT (Devlin *et al.*, 2019) and distilled variants (Sanh *et al.*, 2019), which retain most of BERT's accuracy at a fraction of the inference cost.

## 2.2 Multimodal detection

Text-only analysis is structurally insufficient for the dominant contemporary form of misinformation, in which an authentic image is re-captioned. One lineage fuses ever more expressively, from attention over textual, visual and social features (Jin *et al.*, 2017) to a variational autoencoder's shared latent space (Khattar *et al.*, 2019) and contrastive alignment with optimal transport (Shen *et al.*, 2024). A second treats the image-text *relationship* as the object of study: SAFE scores an explicit similarity (Zhou, Wu and Zafarani, 2020), NewsCLIPpings re-pairs captions with mismatched images (Luo, Darrell and Rohrbach, 2021), and CLIP supplies a ready-made measure of agreement (Radford *et al.*, 2021). Table R1 compares nine such systems by how each is evaluated.

Critically, Papadopoulos *et al.* (2025) argue that leading out-of-context systems learn *similarity*, whether image and caption appear to belong together, rather than *factuality*; similarity correlates with deceptiveness without being equivalent to it, which motivates treating consistency as one explicit input rather than the whole answer. The decisive observation from Table R1 is that eight of the nine systems evaluate on a random split, and the exception holds out events rather than sources.

## 2.3 Large language models as adversary and instrument

Generative models operate in both directions. Zellers *et al.* (2019) showed with Grover that a strong generator of neural fake news is also its strongest detector, and Chen and Shu (2024) note that large models lower the cost of producing coherent misinformation while offering reasoning small classifiers lack. The most directly relevant result is that of Hu *et al.* (2024): a prompted model is a *worse* stand-alone detector than a small fine-tuned model, lacking the domain-specific priors that model absorbs from data, yet the *rationales* it produces contain complementary information, and their Adaptive Rationale Guidance framework has the small model attend to those rationales, outperforming either alone. Vision-language models are now applied directly, with Qi *et al.* (2024) reporting strong out-of-context detection while, importantly here, evaluating on a randomly split synthetic corpus. The lesson carried forward is that a language model is better employed as auxiliary evidence than as the classifier, and that this should be tested rather than assumed.

## 2.4 Benchmarks, distant supervision and the evaluation problem

Benchmark construction has received far less critical attention than architecture design. Fakeddit (Nakamura, Levy and Wang, 2020) labels over a million Reddit submissions by *subreddit membership*: r/photoshopbattles, r/nottheonion and r/upliftingnews are treated as true, and r/propagandaposters, r/savedyouaclick and r/subredditsimulator as varieties of false.

The concern generalises a long-standing result. Torralba and Efros (2011) showed that a classifier can identify which dataset an image came from, so datasets carry a signature independent of content; Gururangan *et al.* (2018) found that inference models could classify hypotheses without the premise, exploiting annotation artefacts; and Geirhos *et al.* (2020) generalise these into shortcut learning, in which models take the easiest predictive feature and identically distributed splits cannot detect it because the shortcut is present on both sides. Bozarth and Budak (2020) show that reported performance is highly sensitive to evaluation design. Nevertheless, as Appendix R records, almost all Fakeddit results use random splits, and if community identity determines the label a random split cannot distinguish a veracity detector from a community detector.

## 2.5 User needs

The users are content-moderation analysts, trust-and-safety engineers and fact-checkers. No primary study with users was undertaken, for the reason in Section 7.4; requirements were elicited instead by structured analysis of regulatory obligations, published accounts of moderation practice, studies of human-AI decision making, and platform enforcement reporting, catalogued in Appendix J. **N1, triage rather than adjudication:** Article 20(6) of the Digital Services Act requires that moderation complaints not be resolved solely by automated means (European Union, 2022), and Nakov *et al.* (2021) characterise the task as assisting rather than replacing human fact-checkers, so the system ranks rather than removes and ranking quality becomes the operative measure. **N2, trustworthy confidence:** Zhang, Liao and Bellamy (2020) show a confidence score improves human-AI decision accuracy only when calibrated and degrades it when not, so expected calibration error (Guo *et al.*, 2017) is reported alongside accuracy. **N3, an inspectable reason:** Das *et al.* (2023) identify explanation as the dominant unmet requirement in human-centred fact-checking and Roberts (2019) documents the volume moderators work under, so a flag a reviewer cannot interrogate wastes their time and threatens legitimate expression. **N4, affordable inference:** enforcement operates at hundreds of millions of actions per quarter (Gillespie, 2018), so per-item cost is a first-class constraint.

## 2.6 Research gap and claimed contribution

Three gaps emerge. Methodologically, the source-leakage hypothesis for Fakeddit has been noted in general terms but never quantified with a content-free probe, a corrected protocol and a decomposition separating source shift from other distribution shift on the same corpus. Architecturally, cross-modal consistency is used either implicitly inside a fusion network or as the entire prediction, rather than as one explicit and inspectable input among several. Empirically, the event-adversarial principle of Wang *et al.* (2018) has not been re-examined as a remedy for *source* leakage.

Three elements are claimed as new: a content-free source probe used as a benchmark diagnostic rather than a baseline; source-disjoint and leave-one-community-out evaluation for distantly supervised corpora, with evidence about what such protocols can and cannot identify; and a consistency gate exposed as a readable per-item quantity. Papadopoulos *et al.* (2025) identify a different confound, in what a model learns; the confound here lies upstream, in label provenance, and applies to any distantly supervised corpus regardless of architecture.

# Chapter 3: Methodology

## 3.1 Description of the artefact

The artefact is an open, end-to-end Python system with four components, shown in Figure 1. A **reproducible corpus pipeline** builds a class-balanced, source-capped corpus with images and materialises three evaluation protocols as columns of one manifest. An **evaluation framework** implements the random split (A), a *source-disjoint* split holding out whole subreddits (B), a temporal split (C), a leave-one-community-out rotation, and a content-free *shortcut probe* whose only input is the source community. **Consistency-Gated Fusion (CGF)** is accompanied by a domain-adversarial variant, four ablations and thirteen baselines from TF-IDF logistic regression to a zero-shot language-model judge; the neural systems train on identical representations with identical optimisation, so differences are attributable to fusion alone. An **analysis layer** produces seed-averaged metrics with bootstrap intervals, corrected significance tests, calibration curves, behavioural probes, measured inference cost and a reviewer triage queue.

![**Figure 1.** Artefact architecture: (a) the pipeline stages and their cached outputs; (b) Consistency-Gated Fusion.](../figures/flow_architecture.png)

## 3.2 Relation to the project aims, and requirements

The aim has two halves, to measure honestly and then to design against the honest measurement, and the artefact is organised around that division: Protocol B and the shortcut probe answer the first (O3), CGF and its adversarial variant the second (O4), and the ablations and behavioural probes establish *why* any difference arises (O6). Seven requirements follow from the objectives and the user needs of Section 2.5: F1 classification from headline, image and metadata; F2 a calibrated probability, not a hard label; F3 an inspectable per-item cross-modal signal; F4 a ranked review queue; NF1 laptop-CPU runtime; NF2 seed reproducibility; NF3 no retained personal data. Appendix G traces each to an acceptance criterion and its verification status.

# Chapter 4: Exploratory Data Analysis

## 4.1 Development methodology

The project followed a hypothesis-driven incremental method, closest to CRISP-DM but with evaluation promoted ahead of modelling, and test-driven at the code level. Appendix K records each cycle, its question and the decision it produced.

Reproducibility was a requirement rather than an aspiration. Every system uses identical hyperparameters, listed with their source in Appendix L, a deliberate trade of per-model tuning for the ability to attribute a difference to the fusion mechanism rather than to a search budget. Twenty-five unit tests (Appendix E) guard the failure modes that would silently invalidate results rather than crash the run, covering split disjointness, standardiser leakage, label polarity, metric correctness and every model's shape contracts.

## 4.2 Corpus construction

The corpus derives from a public mirror of Fakeddit's 100,000-row multimodal split, which retains the full Reddit metadata schema. After removing duplicate image URLs and posts without images, submissions were sampled with a cap of 1,600 per subreddit; capping was preferred to uniform sampling because Fakeddit is dominated by a few large communities, and uniform sampling would leave smaller ones too sparse to hold out under Protocol B. Images were fetched from preview URLs in parallel, of which 96.2% resolved, and the manifest records each row's status so a rebuild is auditable.

The corpus contains 21,798 posts across 17 source communities, 11,305 labelled true and 10,493 false, with nine metadata features available at posting time (Appendix F). Figure 2 shows the property that motivates the study: every community maps to exactly one label.

![**Figure 2.** Every source community maps to one label.](../figures/fig1_corpus_confound.png)

## 4.3 The three evaluation protocols

**Protocol A, stratified random.** A 70/10/20 stratified split matching the published Fakeddit convention, included.

**Protocol B, source-disjoint.** Whole subreddits are assigned to a single split: five communities are held out for test and two for validation, and ten train, so a model that has memorised community style has nothing to memorise about the test communities. This yields 11,806 training, 3,122 validation and 6,870 test posts, and a unit test asserts the disjointness.

**Protocol C, temporal.** Training uses the earliest 70% of posts by creation time and testing the most recent 20%, simulating a deployed detector classifying later content. Source communities persist across time, so this removes temporal but not source leakage.

Protocol B fixes one partition, a selection decision rather than a measurement, so Section 6.4 replaces it with a full rotation.

## 4.4 Tools and rationale

Models are implemented in PyTorch on CPU or Apple MPS. CLIP ViT-B/32 encodes images and supplies the consistency score, its joint image-text space giving a direct agreement measure a vision-only encoder cannot; DistilRoBERTa encodes headlines at 40% of RoBERTa-base's size, cost being a stated requirement; Qwen2.5-0.5B-Instruct is open-weight and CPU-runnable, so no API key is needed; scikit-learn and SciPy supply baselines, metrics and tests; and pytest guards leakage and correctness. Appendix D gives licences and the full rationale.

Two choices merit defence. **Freezing the encoders** trades peak accuracy for throughput and comparability, since the same representations feed every model and an observed difference is therefore attributable to fusion rather than encoder capacity; Section 6.10 tests that claim against an end-to-end fine-tuned baseline rather than asserting it.

## 4.5 Planning, management and control

The project ran under a four-phase plan with fortnightly supervision, covering the literature review and proposal, the corpus pipeline and protocols, the model grid and robustness suite, and the additional analyses and writing; Appendix M compares plan with outcome. That change is recorded as SC-01 in Appendix K: when the shortcut probe returned a perfect score in Phase 2, the project moved from maximising accuracy to measuring and correcting the benchmark, and two planned fusion variants were dropped to fund the source-disjoint protocol, the adversarial variant and the behavioural suite.

# Chapter 5: Implementation and Ethics

## 5.1 System architecture

The pipeline is a chain of deterministic command-line stages, each writing a cached artefact the next consumes (Figure 1a); representations are extracted once and reused, reducing a 165-run grid to about forty minutes of CPU time. Six streams are computed per post: 512-dimensional CLIP visual and text projections, two frozen consistency scores, a 768-dimensional DistilRoBERTa embedding, an 896-dimensional language-model hidden state, and nine standardised metadata features.

## 5.2 Consistency-Gated Fusion

CGF projects the text and image streams into a shared 256-dimensional space, computes interaction terms, concatenates three consistency scores comprising the two frozen measures and a learned cosine in the projected space, and passes the result through a small classifier, as Figure 1(b) shows. Its distinguishing component is a *consistency gate*: a sigmoid of both projected streams and the consistency scores produces a 256-dimensional mask rescaling the visual stream element-wise, so the network can learn to discount vision where the image does not corroborate the claim. Because the gate is a bounded per-example quantity it can be read at inference and inspected, satisfying requirement F3 in a way post-hoc attribution over a concatenation network cannot. The domain-adversarial variant adds a source-community classifier behind a gradient-reversal layer (Ganin *et al.*, 2016) on the fused representation, with the standard ramped weighting; it is a direct response to the confound of Section 4.2 and a re-application of the event-adversarial principle of Wang *et al.* (2018) to *source* rather than event.

## 5.3 The language model as judge and as encoder

Qwen2.5-0.5B-Instruct (Qwen Team, 2024) is used in two ways from a single forward pass. As a **judge**, logits at the answer position are restricted to the two answer tokens and renormalised, giving a zero-shot probability of falsehood without generation, whose calibration is assessed in Section 6.7. As an **encoder**, the final hidden state at that position is retained as a dense feature. Applying the output head at a single position rather than over the whole vocabulary at every position reduced extraction from an estimated eight hours to under three on two CPU cores.

## 5.4 Ethical considerations and their effect on the artefact

Ethics were treated as design constraints with traceable consequences, framed by the BCS Code of Conduct (BCS, 2022). Appendix Q gives the full analysis; four decisions changed the artefact.

**Personal data.** Fakeddit rows carry a Reddit username. Usernames are pseudonymous but re-identifiable and no model requires them, so the build stage drops the `author` column before anything is written to disk, retaining only a binary presence flag: minimisation enacted in code rather than promised in prose. The release ships cached representations rather than images, and because the Fakeddit licence permits research use only and embeddings retain content-derived information, it carries the same restriction and a model card (Mitchell *et al.*, 2019).

**Consent.** Reddit users did not consent to their posts being used for misinformation research. The project processes pseudonymous personal data under the legitimate-interests basis of UK GDPR read with its research provisions, and follows the AoIR position that the ethical weight of public data depends on posters' contextual expectations rather than technical accessibility (franzke *et al.*, 2020). The applicable test is Nissenbaum's (2010) contextual integrity: aggregate methodological analysis does not violate the norms of a public subreddit, whereas reproducing an individual's post beside a machine judgement of its truthfulness would, so no individual post is reproduced here. Fiesler and Proferes (2018) show that user expectations diverge sharply from platform terms, which is why minimisation is enacted in code.

**Labels are proxies, and results are disaggregated.** A post is labelled false because of the community it appeared in, not because a fact-checker assessed it, and treating that proxy as truth is the error this project exists to expose. Because the corpus is English, US-centric and skewed towards 2015 to 2019, a single aggregate would conceal disparities across communities, so Section 6.9 reports each separately.

The pipeline requires roughly five CPU-hours and no GPU, four orders of magnitude below the figures Strubell, Ganesh and McCallum (2019) report for transformer training, which is itself part of the case for freezing the encoders.

**Misuse.** An automated detector is a censorship instrument if used to remove content, so it outputs a calibrated probability with no threshold embedded, calibration error is a first-class metric, and the intended deployment is triage. The project was reviewed under the module's ethics process as secondary analysis of public data with no human participants (Appendix B).

# Chapter 6: Testing

## 6.1 Experimental setup

Twenty-one systems were evaluated under three protocols, the eighteen trainable ones with three seeds each, giving 165 runs. Reported figures are seed-averaged probabilities with 95% percentile bootstrap intervals (Efron and Tibshirani, 1993) over 2,000 resamples; paired comparisons use McNemar's exact test in the form recommended for classifiers (McNemar, 1947; Dietterich, 1998) and a paired bootstrap on the macro-F1 difference, Holm-Bonferroni corrected within protocol across the twenty comparisons made there (Holm, 1979). The positive class is falsehood.

## 6.2 The benchmark is solved by a shortcut

Figure 2 shows that all 17 source communities map to exactly one label. A logistic regression whose *only* input is a one-hot encoding of the subreddit name, with no headline, image or metadata, achieves **100.0% accuracy and 1.000 macro-F1** (95% CI 1.000 to 1.000) on the conventional random split and 0.976 under the temporal split. Against a permutation null in which the community-to-label mapping is destroyed by shuffling training labels, the null centres on 0.408 (SD 0.097, maximum 0.705 over 500 permutations), so the observed score lies far outside it (*p* < 0.01). A perfect detector that has never observed content establishes that a reported accuracy of 90% on Fakeddit is not evidence of misinformation detection. Table 1 reports macro-F1 for every system under all three protocols.

**Table 1:** Macro-F1 by system and protocol, seed-averaged, with 95% bootstrap intervals for the corrected protocol. Two CGF ablations are included because they outperform the full proposed model; Appendix O gives all 21 systems and all metrics.

| System | Random | Source-disjoint (95% CI) | Temporal |
|:--------------------------------|-----------:|:-----------:|-----------:|
| Majority class | 0.342 | 0.288 | 0.344 |
| **Subreddit probe (no content)** | **1.000** | 0.288 | 0.976 |
| LLM zero-shot judge | 0.453 | 0.386 | 0.471 |
| Metadata only | 0.735 | 0.590 | 0.694 |
| Text only (DistilRoBERTa) | 0.790 | 0.617 | 0.639 |
| Image only (CLIP) | 0.808 | 0.624 | 0.641 |
| Early fusion + metadata | 0.900 | 0.649 (0.637-0.661) | 0.720 |
| CGF | 0.905 | 0.625 (0.612-0.636) | 0.743 |
| CGF without the gate | 0.891 | 0.665 (0.653-0.677) | 0.727 |
| **CGF without consistency scores** | 0.890 | **0.679 (0.668-0.690)** | 0.702 |
| CGF + adversarial | 0.892 | 0.653 (0.641-0.664) | 0.730 |
| CGF + LLM + adversarial | 0.891 | 0.672 (0.661-0.684) | 0.721 |

## 6.3 The protocol gap, and what causes it

Under the source-disjoint protocol every system loses a large fraction of its measured performance (Figure 3). The mean reduction across the eighteen learned systems is **0.213 macro-F1**, the best score falls from 0.905 to 0.679, and the ordering changes: Spearman correlation between the two rankings is **rho = 0.68** (*p* = 0.002), and the system ranking first under the random split, CGF without interaction terms, ranks eighth of eighteen under the source-disjoint one. Holding out whole communities also changes the test set's class balance, from 48.1% false to 40.5%, so part of the drop could be prior shift rather than leakage. The source-disjoint test set was therefore resampled without replacement to the random split's prior, giving a matched subsample of 5,786 items on which every system was re-scored. Across the nineteen systems evaluated under both protocols the mean gap is 0.205 before matching and **0.209 after**, so prior shift explains none of it and, if anything, slightly masks it. The gap is attributable to the removal of source information, which is what the protocol was built to remove.

![**Figure 3.** Macro-F1 with 95% intervals, all protocols.](../figures/fig2_main_results.png)

## 6.4 What a single held-out partition cannot tell you

Protocol B holds out one partition of five communities, so its numbers carry a selection uncertainty that item-level bootstrap intervals do not capture. Table 2 reports a full leave-one-community-out rotation, affordable because representations are frozen: each of the fifteen communities large enough to support a fold serves as test in turn, the next in rotation as validation, and the remainder train, so every fold is source-disjoint.

**Table 2:** Leave-one-community-out rotation over fifteen communities, one seed per fold.

| System | Mean accuracy | SD across folds | Worst fold | Best fold |
|:-------------------------|-----------:|-----------:|-----------:|-----------:|
| CGF | 0.611 | 0.245 | 0.193 | 0.979 |
| CGF + adversarial | 0.601 | 0.280 | 0.146 | 0.976 |
| Early fusion + metadata | 0.582 | 0.311 | 0.109 | 0.971 |
| Text only | 0.522 | 0.308 | 0.056 | 0.892 |

This is the most consequential result here. CGF leads the strongest baseline by 0.029 mean accuracy while the standard deviation across folds is 0.245, and on six of fifteen communities CGF is correct on fewer than half the posts. Which community is held out matters roughly ten times more than which architecture is used. The 0.023 macro-F1 by which the proposed system leads on Protocol B's single partition therefore sits inside the noise induced by the choice of partition and is not an identifiable architectural difference.

## 6.5 Ablation: the same component both helps and hurts

Relative to full CGF, removing the consistency scores **improves** source-disjoint macro-F1 by 0.055 but **costs** 0.041 under the temporal protocol; removing the gate improves the source-disjoint score by 0.040 while costing little elsewhere; and removing metadata is detrimental under every protocol, by between 0.019 and 0.048 (Figure 5). Figure 4 accounts for this. CLIP image-headline agreement differs significantly by veracity (0.293 true against 0.257 false, *t* = 55.9, *p* < 0.001, *n* = 21,798) but differs *more* between communities than between classes: r/propagandaposters, labelled false, records among the highest agreement of any community, because propaganda posters are captioned literally. Consistency is genuine within a known community and misleading across communities, so evaluating fusion mechanisms only on random splits favours those exploiting community-specific regularities.

![**Figure 4.** CLIP agreement varies more by community than by veracity.](../figures/fig3_consistency.png)

![**Figure 5.** Ablation, by protocol.](../figures/fig5_ablation.png)

## 6.6 The effect of adversarial training

Penalising source-identifiable representations improves generalisation to unseen communities at little cost elsewhere. The adversary raises CGF from 0.625 to 0.653 macro-F1 under the source-disjoint protocol and CGF+LLM from 0.642 to 0.672, at costs of 0.013 and 0.011 under the random split. Against the strongest non-CGF baseline, early fusion with metadata at 0.649, the full system leads by 0.023 macro-F1 (paired bootstrap 95% CI 0.011 to 0.035), although McNemar on thresholded accuracy is *not* significant for that pair (*p* = 0.60) and Section 6.4 shows the margin to be smaller than partition noise in any case. Calibration improves more convincingly, with expected calibration error of 0.138 against 0.223 for plain CGF and 0.217 for the best baseline (Figure 6), which bears on need N2. The remedy is real but partial, recovering roughly one seventh of the gap.

![**Figure 6.** Reliability diagram, source-disjoint split.](../figures/fig4_calibration_source.png)

## 6.7 The language model: expensive, and useful only indirectly

The zero-shot language-model judge is a poor detector, reaching 0.453 macro-F1 under the random split and 0.386 under the source-disjoint one, below the majority-class baseline on accuracy under the random and temporal splits, with a below-chance AUROC of 0.445 in the source-disjoint condition and a pronounced bias towards predicting falsehood (mean predicted probability 0.756). This reproduces, on a smaller model, the finding of Hu *et al.* (2024). The cost is decisive: on the same two-core CPU, inference requires **427 ms per item** for the language model against **61 ms** for the full CGF pipeline and 0.05 ms for TF-IDF (Figure 7), roughly seven times the cost of everything else combined for a gain of about 0.02 macro-F1. At platform scale this is not defensible under need N4, and it is the class of evidence accuracy-only comparisons cannot supply.

![**Figure 7.** Macro-F1 against measured inference cost.](../figures/fig8_cost.png)

## 6.8 Behavioural and robustness testing

Accuracy does not establish that a model uses the image-headline relationship, whereas a behavioural test does. Every test image was re-paired with another post's image, headlines intact. CGF responds most strongly, with a mean change in predicted falsehood of +0.231 against +0.181 for the image-only model and +0.177 for early fusion, while the text-only model, correctly, does not respond. Gate inspection is less favourable: it correlates only weakly with CLIP consistency (*r* = 0.19, Figure 8) and its dynamic range is narrow. Appendix O gives the full tables.

![**Figure 8.** Visual gate against CLIP consistency.](../figures/fig7_gate_source.png)

## 6.9 Ranking quality, the triage queue and per-community performance

Need N1 makes ranking the operative task, so precision at the top of the queue is the metric the use case demands. Against a prevalence of 0.405 in the source-disjoint test set, early fusion with metadata achieves the highest precision at 100 items, 0.86, a lift of 2.12; CGF reaches 0.82 and the full proposed system 0.75, while the subreddit probe and the text-only model collapse to chance at 0.40 and the language-model judge to 0.25. The ordering by ranking quality differs from the ordering by macro-F1 and the proposed system is not first on it, so reporting only macro-F1 would have set a requirement in Chapter 2 and measured something else.

The artefact produces the queue itself, not merely the metric: each row carries the calibrated probability, the empirical precision of its score band, the CLIP consistency and the gate activation, so a reviewer sees both the ranking and the evidence for it (Appendix P). Section 5.4's commitment to disaggregated reporting is discharged here: across the five held-out communities the proposed system's accuracy ranges from 0.472 on r/confusing_perspective to 0.841 on r/usnews, a spread sixteen times the 0.023 margin separating the leading systems, so a deployment decision taken on the aggregate would rest on a number describing no individual community.

## 6.10 Error analysis, the frozen-encoder check and threats to validity

The frozen-encoder decision was checked directly. Fine-tuning DistilRoBERTa end to end on the same source-disjoint split, 82 million trainable parameters and 26 minutes of CPU training against roughly eight seconds for the frozen head, reaches 0.604 macro-F1, *below* the 0.617 of the frozen text embedding with a small MLP. Encoder capacity is therefore not what limits performance under source shift; the protocol is. Per-category accuracy is uneven, clickbait misrepresenting a linked article being hardest (Appendix H). Two threats temper the conclusions. The shortcut is demonstrated on a **single benchmark**, so the proposition that similar confounds affect other distantly supervised corpora is an inference from the dataset-bias literature rather than a measured result. The rotation of Section 6.4 uses **one seed per fold**, so part of the reported spread is seed variance, although the seed variance measured under Protocol B is 0.008 macro-F1, negligible against a fold spread of 0.245.

# Chapter 7: Discussion, Evaluation of the Approach and Reflection

## 7.1 Interpretation of the results

The principal claim is not that CGF is a better detector but that the field's measurements do not measure what they purport to. Three findings support it: the subreddit probe scores 1.000 without observing content, the ranking of systems changes materially between protocols (rho = 0.68), and the gap survives intact when class-prior shift is removed.

A second finding is more useful to practitioners: **component value is protocol-dependent**. Cross-modal consistency is worth 0.041 macro-F1 within known communities and costs 0.055 across unknown ones, so a design decision cannot be evaluated in the abstract, only against a deployment assumption.

The third was unexpected and would not have been found without the rotation of Section 6.4. Removing the shortcut does not make architectural comparison possible; it reveals that the comparison was never identifiable at this corpus size, since a margin of 0.023 macro-F1 sits inside a fold-to-fold standard deviation of 0.245. Claims of architectural superiority on a single held-out partition of a distantly supervised corpus are therefore not supported by the available evidence, this project's own included.

## 7.2 Extent to which the aims and objectives were met

The first half of the aim, quantifying how much reported performance survives an honest protocol, is met without qualification: a gap of 0.213 macro-F1 is measured, tested against a prior-shift alternative and significance-tested with correction. The second, designing a detector that improves on baselines under that protocol, is **not** met. The full system leads the strongest baseline by 0.023 macro-F1 with a bootstrap interval excluding zero, but with no significant McNemar result, with two of CGF's own ablations ahead of it under the corrected protocol, with lower precision at 100 than a simpler baseline, and with a margin an order of magnitude smaller than the partition noise of Section 6.4. The proposed architecture is clearly best under the contaminated protocol and indistinguishable from simpler alternatives under the clean one, which is this report's own argument applied to its own contribution.

Against the criteria in Appendix S, O2 to O7 were met; O1 reached 30 sources only after literature added in Phase 4, the original review having reached 25, and recording that shortfall matters in a report arguing for honest measurement. The problem itself is not solved: a system incorrect on a third of posts from unseen communities cannot be deployed autonomously.

## 7.3 Evaluation of the general approach

The approach, comprising a single benchmark, a content-free probe, three protocols with a rotation, frozen encoders and a purely quantitative design, deserves the same scrutiny as the artefact.

Its strength is that two central decisions were tested rather than assumed: freezing the encoders cost no accuracy under source shift when checked against a fine-tuned control (Section 6.10), and the protocol gap survived a check against the obvious alternative explanation, prior shift (Section 6.3).

Its principal weakness is that the instrument is coarser than the question. Protocol B answers "is performance lower when sources are unseen" convincingly, but the rotation shows it cannot answer "which architecture generalises better": fifteen folds on 21,798 posts leave between-model differences an order of magnitude inside fold variance, and recognising this changed what the project claims.

## 7.4 Reflection

The shortcut probe should have been the *first* experiment. Effort went into fusion architecture before it reframed the project, and although little of that code was wasted, the framing effort was; building the cheapest falsification test before the expensive constructive work is a lesson that generalises. Two further choices would be made differently: requirements would be derived from a small number of interviews rather than documents alone, and the leave-one-community-out rotation would have been the primary protocol from the outset, since it is what actually determined the conclusion.

Freezing the encoders proved the most productive decision. It appeared at the time to be a compromise, and it is why 165 training runs and a fifteen-fold rotation fit on a laptop, which made the ablation, the protocol comparison and the rotation affordable. The most valuable habit was verifying every reported number against the artefact's output files rather than transcribing it, which caught four rounding errors and a misstatement of which system was best.

# Chapter 8: Conclusion

This project asked how much of the reported performance of multimodal misinformation detectors survives an evaluation protocol that removes source leakage. The artefact answers directly: a reproducible pipeline over 21,798 Fakeddit posts, twenty-one systems, three protocols with a leave-one-community-out rotation, and an analysis layer with uncertainty quantification and corrected significance testing.

Roughly a quarter of the measured macro-F1 does not survive. A classifier observing only the source community's name achieves 100.0% accuracy on the conventional random split against a permutation null of 0.408, establishing that Fakeddit's distantly supervised labels are a deterministic function of source. Holding out whole communities costs the average system 0.213 macro-F1, a loss that class-prior matching shows is not an artefact of changed class balance, and reorders which architecture appears best (rho = 0.68). Cross-modal consistency helps within known communities and hurts across unknown ones; adversarial training recovers about one seventh of the gap and cuts expected calibration error from 0.223 to 0.138; and a 0.5B language model is a below-chance judge at seven times the pipeline's cost.

The most important result limits the project's own contribution. A fifteen-fold community rotation shows generalisation to an unseen community varying by a standard deviation of 0.245 accuracy, against 0.029 separating the best and worst architectures tested. The aim's measurement half is met; its design half is not, because on this corpus that comparison is not identifiable. Correcting a benchmark's split reveals the shortcut but does not by itself deliver a protocol on which architectures can be ranked. Three lines of future work follow: **leave-one-community-out at scale**, across several corpora and seeds, to establish the sample size at which architectural comparison becomes identifiable; **evidence-grounded detection** after Thorne and Vlachos (2018); and **replication of the shortcut probe** on other distantly supervised corpora. At roughly 69% accuracy on unseen communities, automated detection belongs in a human-supervised triage pipeline, not an autonomous role.

# Bibliography

Alam, F., Cresci, S., Chakraborty, T., Silvestri, F., Dimitrov, D., Da San Martino, G., Shaar, S., Firooz, H. and Nakov, P. (2022) ‘A survey on multimodal disinformation detection’, Proceedings of the 29th International Conference on Computational Linguistics (COLING 2022), pp. 6625–6643. Available from: https://aclanthology.org/2022.coling-1.576/. [Accessed 1 September 2026].

BCS, The Chartered Institute for IT (2022) Code of Conduct for BCS Members [online]. Version 8. Swindon: BCS. Available from: https://www.bcs.org/media/2211/bcs-code-of-conduct.pdf [Accessed 29 August 2026].
Bozarth, L. and Budak, C. (2020) ‘Toward a better performance evaluation framework for fake news classification’, Proceedings of the International AAAI Conference on Web and Social Media, 14(1), pp. 60–71. Available from: https://ojs.aaai.org/index.php/ICWSM/article/view/7279 . [Accessed 27 August 2026].

Chen, C. and Shu, K. (2024) ‘Combating misinformation in the age of LLMs: opportunities and challenges’, AI Magazine [online], 45(3), pp. 354–368. Available from: https://doi.org/10.1002/aaai.12188 . [Accessed 3 September 2026].
Das, A., Liu, H., Kovatchev, V. and Lease, M. (2023) ‘The state of human-centered NLP technology for fact-checking’, Information Processing & Management [online]. 60(2), 103219. Available from: https://doi.org/10.1016/j.ipm.2022.103219 . [Accessed 25 August 2026]

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) ‘BERT: pre-training of deep bidirectional transformers for language understanding’, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186. Available from: https://aclanthology.org/N19-1423/ . [Accessed 1 September 2026].
Dietterich, T.G. (1998) ‘Approximate statistical tests for comparing supervised classification learning algorithms’, Neural Computation [online].10(7), pp. 1895–1923. Available from: https://doi.org/10.1162/089976698300017197 . [Accessed 2 September 2026].

Efron, B. and Tibshirani, R.J. (1993) An Introduction to the Bootstrap. Monographs on Statistics and Applied Probability 57. New York: Chapman & Hall. [online]. Available from: https://doi.org/10.1201/9780429246593 . [Accessed 2 September 2026].
European Union (2022) Regulation (EU) 2022/2065 of the European Parliament and of the Council of 19 October 2022 on a Single Market For Digital Services and amending Directive 2000/31/EC (Digital Services Act), Official Journal of the European Union, L 277, 210.2022, p. 1. Available from: https://eur-lex.europa.eu/eli/reg/2022/2065/oj/eng . [Accessed 21 August 2026]

Fiesler, C. and Proferes, N. (2018) ‘“Participant” perceptions of Twitter research ethics’, Social Media + Society, 4(1), 2056305118763366. Available from: https://doi.org/10.1177/2056305118763366 . [Accessed 21 August 2026]

Franzke, a. s., Bechmann, A., Zimmer, M., Ess, C. and the Association of Internet Researchers (2020) Internet Research: Ethical Guidelines 3.0 [online]. Association of Internet Researchers. Available from: https://aoir.org/reports/ethics3.pdf [Accessed 22 August 2026]

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M. and Lempitsky, V. (2016) ‘Domain-adversarial training of neural networks’, Journal of Machine Learning Research [online].17(59), pp. 1–35. Available from: https://www.jmlr.org/papers/v17/15-239.html [Accessed 27 August 2026]

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M. and Wichmann, F.A. (2020) ‘Shortcut learning in deep neural networks’. Nature Machine Intelligence [online]. 2 (11), pp. 665–673. Available from: https://doi.org/10.1038/s42256-020-00257-z [Accessed 3 September 2026].

Gillespie, T. (2018) Custodians of the Internet: Platforms, Content Moderation, and the Hidden Decisions That Shape Social Media [online]. New Haven, CT: Yale University Press. Available from: https://yalebooks.yale.edu/book/9780300261431/custodians-of-the-internet/ [Accessed 1 September 2026]. 

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) ‘On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning. Sydney, Australia, 6–11 August 2017. Proceedings of Machine Learning Research, 70, pp. 1321–1330. Available from: https://proceedings.mlr.press/v70/guo17a.html [Accessed 2 September 2026]. 

Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S.R. and Smith, N.A. (2018) ‘Annotation artifacts in natural language inference data’. In: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers). New Orleans, Louisiana, 1–6 June 2018. Association for Computational Linguistics, pp. 107–112. Available from: https://aclanthology.org/N18-2017/ [Accessed 3 September 2026].

Holm, S. (1979) ‘A simple sequentially rejective multiple test procedure’. Scandinavian Journal of Statistics [online]. 6 (2), pp. 65–70. Available from: https://www.jstor.org/stable/4615733 [Accessed 28 August 2026].

Hu, B., Sheng, Q., Cao, J., Shi, Y., Li, Y., Wang, D. and Qi, P. (2024) ‘Bad actor, good advisor: exploring the role of large language models in fake news detection’. In: Proceedings of the AAAI Conference on Artificial Intelligence. Vancouver, Canada, 20–27 February 2024. Association for the Advancement of Artificial Intelligence, 38 (20), pp. 22105–22113. Available from: https://ojs.aaai.org/index.php/AAAI/article/view/30214 [Accessed 3 September 2026] 

Jin, Z., Cao, J., Guo, H., Zhang, Y. and Luo, J. (2017) ‘Multimodal fusion with recurrent neural networks for rumor detection on microblogs. In: Proceedings of the 25th ACM International Conference on Multimedia. Mountain View, California, 23–27 October 2017. Association for Computing Machinery, pp. 795–816. Available from: https://doi.org/10.1145/3123266.3123454  [Accessed 25 August 2026].

Khattar, D., Goud, J.S., Gupta, M. and Varma, V. (2019) ‘MVAE: multimodal variational autoencoder for fake news detection’. In: Proceedings of the 2019 World Wide Web Conference (WWW ’19). San Francisco, California, 13–17 May 2019. Association for Computing Machinery, pp. 2915–2921. Available from: https://doi.org/10.1145/3308558.3313552 . [Accessed 3 September 2026].

Luo, G., Darrell, T. and Rohrbach, A. (2021) ‘NewsCLIPpings: automatic generation of out-of-context multimodal media’. In: Moens, M.-F., Huang, X., Specia, L. and Yih, S.W.-t., eds. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Online and Punta Cana, Dominican Republic, 7–11 November 2021. Association for Computational Linguistics, pp. 6801–6817. Available from: https://aclanthology.org/2021.emnlp-main.545/ .[Accessed 18 August 2026].

McNemar, Q. (1947) ‘Note on the sampling error of the difference between correlated proportions or percentages’. Psychometrika [online]. 12 (2), pp. 153–157. Available from: https://doi.org/10.1007/BF02295996 .[Accessed 18 August 2026].

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D. and Gebru, T. (2019) ‘Model cards for model reporting’. In: FAT ’19: Proceedings of the Conference on Fairness, Accountability, and Transparency. Atlanta, Georgia, 29–31 January 2019. Association for Computing Machinery, pp. 220–229. Available from: https://doi.org/10.1145/3287560.3287596 .[Accessed 20 August 2026].

Nakamura, K., Levy, S. and Wang, W.Y. (2020) ‘Fakeddit: a new multimodal benchmark dataset for fine-grained fake news detection’. In: Proceedings of the Twelfth Language Resources and Evaluation Conference. Marseille, France, 11–16 May 2020. European Language Resources Association, pp. 6149–6157. Available from: https://aclanthology.org/2020.lrec-1.755/ . [Accessed 20 August 2026].

Nakov, P., Corney, D., Hasanain, M., Alam, F., Elsayed, T., Barrón-Cedeño, A., Papotti, P., Shaar, S. and Da San Martino, G. (2021) ‘Automated fact-checking for assisting human fact-checkers’. In: Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21). Montreal, Canada, 19–27 August 2021. International Joint Conferences on Artificial Intelligence Organization, pp. 4551–4558. Available from: https://www.ijcai.org/proceedings/2021/619. [Accessed 30 August 2026]. 

Nissenbaum, H. (2010) Privacy in Context: Technology, Policy, and the Integrity of Social Life [online]. Stanford, CA: Stanford University Press. Available from: https://www.sup.org/books/law/privacy-context . [Accessed 3 September 2026]. 

Papadopoulos, S.-I., Koutlis, C., Papadopoulos, S. and Petrantonakis, P.C. (2025) ‘Similarity over factuality: are we making progress on multimodal out-of-context misinformation detection?’, Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025), pp. 5570–5579. Available from: https://openaccess.thecvf.com/content/WACV2025/html/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.html . [Accessed 1 September 2026].

Qi, P., Yan, Z., Hsu, W. and Lee, M.L. (2024) ‘SNIFFER: multimodal large language model for explainable out-of-context misinformation detection’, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024), pp. 13052–13062. Available from:  https://openaccess.thecvf.com/content/CVPR2024/html/Qi_SNIFFER_Multimodal_Large_Language_Model_for_Explainable_Out-of-Context_Misinformation_Detection_CVPR_2024_paper.html .[Accessed 1 September 2026].

Qwen Team (2024) Qwen2.5 technical report [online]. arXiv:2412.15115. Available from: https://arxiv.org/abs/2412.15115 .[Accessed 1 September 2026].

Papadopoulos, S.-I., Koutlis, C., Papadopoulos, S. and Petrantonakis, P.C. (2025) Similarity over factuality: are we making progress on multimodal out-of-context misinformation detection? In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025), pp. 5570–5579. Available from: https://openaccess.thecvf.com/content/WACV2025/html/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.html .[Accessed 1 September 2026].

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G. and Sutskever, I. (2021) ‘Learning transferable visual models from natural language supervision’, Proceedings of the 38th International Conference on Machine Learning, 139, pp. 8748–8763. Available from: https://proceedings.mlr.press/v139/radford21a.html .[Accessed 2 September 2026].

Hu, B., Sheng, Q., Cao, J., Shi, Y., Li, Y., Wang, D. and Qi, P. (2024) Bad actor, good advisor: exploring the role of large language models in fake news detection. In: Proceedings of the AAAI Conference on Artificial Intelligence, 38 (20), pp. 22105–22113. Available from: https://ojs.aaai.org/index.php/AAAI/article/view/30214 .[Accessed 2 September 2026].

Rashkin, H., Choi, E., Jang, J.Y., Volkova, S. and Choi, Y. (2017) ‘Truth of varying shades: analyzing language in fake news and political fact-checking’, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 2931–2937. Available from: https://aclanthology.org/D17-1317/ .[Accessed 10 August 2026].

Roberts, S.T. (2019) Behind the Screen: Content Moderation in the Shadows of Social Media [online]. New Haven, CT: Yale University Press. Available from: https://yalebooks.yale.edu/book/9780300261479/behind-the-screen/ .[Accessed 20 August 2026].

Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019) DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108. Available from: https://arxiv.org/abs/1910.01108 .[Accessed 30 August 2026].

Shen, X., Huang, M., Hu, Z., Cai, S. and Zhou, T. (2024) ‘Multimodal fake news detection with contrastive learning and optimal transport’, Frontiers in Computer Science, 6, article 1473457. Available from: https://doi.org/10.3389/fcomp.2024.1473457 .[Accessed 25 August 2026].

Shu, K., Sliva, A., Wang, S., Tang, J. and Liu, H. (2017) ‘Fake news detection on social media: a data mining perspective’, ACM SIGKDD Explorations Newsletter [online]. 19(1), pp. 22–36. Available from: https://doi.org/10.1145/3137597.3137600 .[Accessed 30 August 2026].

Spearman, C. (1904) ‘The proof and measurement of association between two things’. The American Journal of Psychology [online]. 15 (1), pp. 72–101. Available from: https://doi.org/10.2307/1412159 .[Accessed 2 September 2026].

Strubell, E., Ganesh, A. and McCallum, A. (2019) ‘Energy and policy considerations for deep learning in NLP’, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 3645–3650. Available from: https://aclanthology.org/P19-1355/ .[Accessed 1 September 2026]. 

Thorne, J. and Vlachos, A. (2018) ‘Automated fact checking: task formulations, methods and future directions. In: Bender, E.M., Derczynski, L. and Isabelle, P., eds. Proceedings of the 27th International Conference on Computational Linguistics. Santa Fe, New Mexico, USA, 20–25 August 2018. Association for Computational Linguistics, pp. 3346–3359. Available from: https://aclanthology.org/C18-1283/ .[Accessed 1 September 2026]. 

Torralba, A. and Efros, A.A. (2011) ‘Unbiased look at dataset bias’, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2011), pp. 1521–1528. Available from: https://doi.org/10.1109/CVPR.2011.5995347 .[Accessed 12 August 2026]. 

Wang, Y., Ma, F., Jin, Z., Yuan, Y., Xun, G., Jha, K., Su, L. and Gao, J. (2018) ‘EANN: event adversarial neural networks for multi-modal fake news detection’, Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 849–857. Available from: https://doi.org/10.1145/3219819.3219903 .[Accessed 1 September 2026]. 

Zellers, R., Holtzman, A., Rashkin, H., Bisk, Y., Farhadi, A., Roesner, F. and Choi, Y. (2019) ‘Defending against neural fake news’, Advances in Neural Information Processing Systems 32 (NeurIPS 2019). Available from: https://proceedings.neurips.cc/paper/2019/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html .[Accessed 1 September 2026].

Shen, X., Huang, M., Hu, Z., Cai, S. and Zhou, T. (2024) ‘Multimodal fake news detection with contrastive learning and optimal transport’, Frontiers in Computer Science, 6, article 1473457. Available from: https://doi.org/10.3389/fcomp.2024.1473457 .[Accessed 29 August 2026].

Zhang, Y., Liao, Q.V. and Bellamy, R.K.E. (2020) ‘Effect of confidence and explanation on accuracy and trust calibration in AI-assisted decision making’, Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT* ’20), pp. 295–305. Available from: https://doi.org/10.1145/3351095.3372852 .[Accessed 29 August 2026].

Zhou, X. and Zafarani, R. (2020) ‘A survey of fake news: fundamental theories, detection methods, and opportunities’, ACM Computing Surveys [online]. 53(5), article 109, pp. 1–40. Available from: https://doi.org/10.1145/3395046 .[Accessed 28 August 2026].

Zhou, X., Wu, J. and Zafarani, R. (2020) ‘SAFE: similarity-aware multi-modal fake news detection’, Advances in Knowledge Discovery and Data Mining (PAKDD 2020), Lecture Notes in Computer Science, 12085, pp. 354–367. Available from: https://doi.org/10.1007/978-3-030-47436-2_27 .[Accessed 27 August 2026].

