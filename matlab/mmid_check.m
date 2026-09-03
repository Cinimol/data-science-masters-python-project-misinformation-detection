function ok = mmid_check()
%MMID_CHECK  Diagnose the environment before running the figure scripts.
%
%   Run this first if a figure script fails. It reports, in order, the four
%   things that actually go wrong: the helper functions not being on the path,
%   the exported plotting data being absent, an individual data file being
%   missing or unreadable, and the output directory not being writable. Each
%   failure prints the command that fixes it rather than a stack trace.
%
%       >> cd matlab
%       >> mmid_check
%
%   Returns true when every check passes.

    fprintf('mmid figure suite: environment check\n');
    fprintf('------------------------------------\n');
    ok = true;

    here = fileparts(mfilename('fullpath'));
    if ~any(strcmp(here, regexp(path, pathsep, 'split')))
        addpath(here);
        fprintf('[fixed]  added %s to the MATLAB path\n', here);
    end
    fprintf('[ok]     scripts found in %s\n', here);

    if exist('OCTAVE_VERSION', 'builtin')
        fprintf('[ok]     GNU Octave %s\n', version());
    else
        fprintf('[ok]     MATLAB %s\n', version('-release'));
    end

    helpers = {'mmid_style', 'mmid_readcsv'};
    for k = 1:numel(helpers)
        if exist(helpers{k}, 'file') ~= 2
            fprintf('[FAIL]   %s.m not found. Run this from the matlab/ ');
            fprintf('directory, or add it to the path.\n', helpers{k});
            ok = false;
        end
    end
    if ~ok
        return
    end

    dataDir = mmid_style('datadir');
    if exist(dataDir, 'dir') ~= 7
        fprintf(['[FAIL]   no data directory at %s\n' ...
                 '         Fix: run "python -m src.export_matlab" at the ' ...
                 'repository root.\n'], dataDir);
        ok = false;
        return
    end
    fprintf('[ok]     data directory %s\n', dataDir);

    needed = {'fig1_subreddit_counts.csv', 'fig2_main_results.csv', ...
              'fig3_item_consistency.csv', 'fig3_subreddit_consistency.csv', ...
              'fig4_calibration_random.csv', 'fig4_calibration_source.csv', ...
              'fig4_calibration_temporal.csv', 'fig5_ablation.csv', ...
              'fig6_error_by_category_random.csv', ...
              'fig6_error_by_category_source.csv', ...
              'fig6_error_by_category_temporal.csv', ...
              'fig7_gate_random.csv', 'fig7_gate_source.csv', ...
              'fig7_gate_temporal.csv', 'fig8_cost.csv'};
    missing = {};
    for k = 1:numel(needed)
        f = fullfile(dataDir, needed{k});
        if exist(f, 'file') ~= 2
            missing{end + 1} = needed{k};                        %#ok<AGROW>
            continue
        end
        try
            S = mmid_readcsv(f);
            flds = fieldnames(S);
            rows = numel(S.(flds{1}));
            fprintf('[ok]     %-36s %5d rows, %d columns\n', ...
                    needed{k}, rows, numel(flds));
        catch err
            fprintf('[FAIL]   %s could not be read: %s\n', needed{k}, ...
                    err.message);
            ok = false;
        end
    end
    if ~isempty(missing)
        fprintf('[FAIL]   %d data files missing:\n', numel(missing));
        fprintf('           %s\n', missing{:});
        fprintf(['         Fix: run "python -m src.export_matlab" at the ' ...
                 'repository root.\n']);
        ok = false;
    end

    outDir = mmid_style('outdir');
    probe = fullfile(outDir, 'mmid_write_probe.tmp');
    fid = fopen(probe, 'w');
    if fid < 0
        fprintf('[FAIL]   cannot write to %s\n', outDir);
        ok = false;
    else
        fclose(fid);
        delete(probe);
        fprintf('[ok]     output directory %s is writable\n', outDir);
    end

    fprintf('------------------------------------\n');
    if ok
        fprintf('All checks passed. Run: run_all_figures\n');
    else
        fprintf('Fix the items marked FAIL above, then run mmid_check again.\n');
    end
end
