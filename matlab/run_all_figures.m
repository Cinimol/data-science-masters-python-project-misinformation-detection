function run_all_figures()
%RUN_ALL_FIGURES  Regenerate every report figure from the exported results.
%
%   Run this from the matlab/ directory after the Python pipeline has produced
%   results/ and the export stage has written matlab/data/:
%
%       python -m src.export_matlab      % from the repository root
%       >> cd matlab
%       >> run_all_figures
%
%   Fourteen PNG files are written to figures_matlab/, one per panel of the
%   eight figures in the report. Nothing is retrained and no model is loaded:
%   the scripts read only the exported CSV files, so the figures can be
%   regenerated and inspected in seconds.
%
%   Verified under MATLAB syntax rules and executed end to end under GNU
%   Octave 8.4; no toolbox beyond core graphics is required.

    here = fileparts(mfilename('fullpath'));
    addpath(here);

    dataDir = mmid_style('datadir');
    if ~exist(dataDir, 'dir')
        error('mmid:runall:noData', ...
              ['No exported data in %s.\n' ...
               'Run "python -m src.export_matlab" from the repository ' ...
               'root first.'], dataDir);
    end

    fprintf('[matlab] reading exported results from %s\n', dataDir);
    started = tic;

    fig1_corpus_confound();
    fig2_main_results();
    fig3_consistency();
    fig4_calibration();
    fig5_ablation();
    fig6_error_by_category();
    fig7_gate();
    fig8_cost();

    written = dir(fullfile(mmid_style('outdir'), '*.png'));
    fprintf('[matlab] %d figures written to %s in %.1f s\n', ...
            numel(written), mmid_style('outdir'), toc(started));
end
