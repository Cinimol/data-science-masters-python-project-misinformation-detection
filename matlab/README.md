# MATLAB figure suite

A MATLAB implementation of every figure in the report. The scripts read only
the CSV files in `matlab/data/`, so the figures can be regenerated and adjusted
in seconds without loading a model, retraining anything, or having Python
installed at all.

## 1. Running it

From the repository root, export the plotting data once (this is the only step
that needs Python, and it re-reads the pipeline's cached results rather than
recomputing them):

```bash
python -m src.export_matlab       # or: make matlab-data
```

Then, in MATLAB:

```matlab
cd matlab
run_all_figures
```

Fourteen PNG files are written to `figures_matlab/`, one per panel of the eight
report figures. Each script can also be run on its own, and the three that vary
by evaluation protocol accept the protocol as an argument:

```matlab
fig2_main_results
fig4_calibration('source')
fig6_error_by_category('temporal')
r = fig7_gate('source');          % returns the gate/consistency correlation
```

If anything fails, run the environment check first. It reports the four things
that actually go wrong, in the order they occur, and prints the command that
fixes each one rather than a stack trace:

```matlab
cd matlab
mmid_check
```

It confirms that the helper functions are on the path, names the interpreter
and release in use, reads every expected CSV and reports its row and column
count, and probes that `figures_matlab/` is writable. The most common failure
is the plotting data never having been exported, which shows as fifteen missing
files and the `python -m src.export_matlab` fix.

## 2. Files

| File | Figure |
|:-----|:-------|
| `fig1_corpus_confound.m` | Figure 1: every subreddit maps to exactly one label |
| `fig2_main_results.m` | Figure 2: macro-F1 for all systems under all three protocols |
| `fig3_consistency.m` | Figure 3: CLIP agreement by class and by community |
| `fig4_calibration.m` | Figure 4: reliability diagrams |
| `fig5_ablation.m` | Figure 5: component ablation |
| `fig6_error_by_category.m` | Figure 6: accuracy by misinformation category |
| `fig7_gate.m` | Figure 7: learned gate against cross-modal consistency |
| `fig8_cost.m` | Figure 8: accuracy against measured inference cost |
| `run_all_figures.m` | Driver that calls all eight in order |
| `mmid_style.m` | Shared palette, axis styling, legend and handle helpers, PNG export |
| `mmid_readcsv.m` | Minimal CSV reader used by every script |
| `mmid_check.m` | Environment diagnostic, run this first if a script fails |

## 3. Requirements

Core MATLAB only. No toolbox is used: the CSV reader is built on `fileread`,
`regexp` and `sscanf` rather than `readtable` or `textscan`, histograms are
computed with `histc` rather than `histogram`, and figures are written with
`print` rather than `exportgraphics`, so the suite runs on old releases as well
as current ones.

Three portability points are handled centrally in `mmid_style.m` rather than
being repeated in the figure scripts, because each of them fails on one
interpreter and not the other:

- **Handle arrays.** MATLAB R2014b and later return graphics objects from
  `bar` and `plot`, so assigning one into an array preallocated with `zeros`
  raises *value of type Bar is not convertible to double*. Octave still uses
  numeric handles, where `gobjects` does not exist. `mmid_style('handles', n)`
  allocates whichever is correct.
- **Legend properties.** Passing `Box` or `FontSize` directly to `legend`
  alongside a location string is accepted by some releases and rejected by
  others. `mmid_style('legend', ...)` creates the legend first and then assigns
  each property, guarded, so an unsupported one degrades the appearance instead
  of aborting the figure.
- **Tick label interpreters.** Octave's gnuplot backend accepts
  `TickLabelInterpreter` and then ignores it, which would draw the underscore
  in `confusing_perspective` as a subscript. On that backend only, the label
  text is escaped instead; MATLAB honours the property and is left alone.

The suite was executed end to end under GNU Octave 8.4, where all fourteen PNGs
are produced in under twenty seconds. The three points above are the places
where MATLAB and Octave genuinely differ, and each is resolved by asking the
interpreter what it supports rather than by assuming one of them.

## 4. Relationship to the Python figures

`src/analyse.py` remains the authoritative implementation: it produces
`figures/*.png`, which are the figures embedded in the report. The MATLAB suite
redraws the same quantities from the same numbers, and `src/export_matlab.py`
is a pure flattening step rather than a second analysis, so the two cannot
disagree. The gate/consistency correlation is computed independently on each
side and agrees to machine precision, which is the check that the port reads
the exported arrays correctly.
