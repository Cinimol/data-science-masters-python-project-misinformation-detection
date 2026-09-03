PY ?= python

.PHONY: all smoke data features llm train robustness analyse extras diagrams matlab-data report test clean

all: data features llm train robustness analyse extras diagrams

## Stage 1 -- build the corpus (network bound, ~15 min)
data:
	$(PY) -m src.build_dataset

## Stage 2 -- frozen CLIP + DistilRoBERTa representations
features:
	$(PY) -m src.features --stages base

## Stage 2b -- LLM hidden states and zero-shot judgements (optional, slow)
llm:
	$(PY) -m src.features --stages llm

## Stage 4 -- the full experiment grid
train:
	$(PY) -m src.train

## Stage 6 -- robustness, behavioural probes, inference cost
robustness:
	$(PY) -m src.robustness

## Stage 5 -- aggregation, significance tests, figures
analyse:
	$(PY) -m src.analyse

## End-to-end fine-tuned text baseline (optional, ~40 min CPU)
finetune:
	$(PY) -m src.finetune_text --protocol source --epochs 2
	$(PY) -m src.finetune_text --protocol random --epochs 2

## Five-minute smoke test of the whole path on a tiny sample
smoke:
	$(PY) -m src.build_dataset --limit 400 --out smoke.parquet
	$(PY) -m src.features --corpus smoke.parquet --prefix smoke --stages base
	$(PY) -m src.train --corpus smoke.parquet --prefix smoke \
		--protocols random --models text,concat,cgf --seeds 42 --out smoke_runs.csv

test:
	$(PY) -m pytest tests/ -q

## Score one image/headline pair with the trained artefact
## Usage: make predict IMAGE=path/to/photo.jpg HEADLINE="a claimed headline"
predict:
	$(PY) -m src.predict --image "$(IMAGE)" --headline "$(HEADLINE)"

clean:
	rm -rf results/preds results/*.csv results/*.json figures/*.png

## Stage 9 -- flatten the figure data into CSV for the MATLAB suite
## Stage 10-12 -- rotation, probe null, ranking, architecture figure, triage
## Stage 13 -- flowcharts as draw.io sources and figures
diagrams:
	$(PY) -m src.flowcharts

## Stage 10-12 -- rotation, probe null, ranking, architecture figure, triage
extras:
	$(PY) -m src.extra_analyses
	$(PY) -m src.fig_architecture
	$(PY) -m src.triage

## Stage 9 -- flatten the figure data into CSV for the MATLAB suite
matlab-data:
	$(PY) -m src.export_matlab

## Build the report .docx from Markdown and apply the house table style
report:
	cd docs && pandoc report.md appendices.md -o "report.docx" \
		--toc --toc-depth=2 --resource-path=.:..
	$(PY) -m src.style_docx "docs/report.docx"

## Interactive menu -- the easy way in
run:
	$(PY) run.py
