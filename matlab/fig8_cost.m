function fig8_cost()
%FIG8_COST  Figure 8: source-disjoint macro-F1 against inference cost.
%
%   Cost is measured per item on two CPU cores and plotted on a logarithmic
%   axis spanning five orders of magnitude. A detector that cannot run at
%   platform scale is not a deployable answer to the research question, so
%   accuracy and cost belong on the same axes. The language-model variants sit
%   far to the right for a modest and, under the corrected protocol, largely
%   insignificant gain.
%
%   Input : data/fig8_cost.csv
%   Output: figures_matlab/fig8_cost.png

    S = mmid_readcsv(fullfile(mmid_style('datadir'), 'fig8_cost.csv'));
    col = mmid_style('colours');

    % Label offsets in points, tuned so that no two annotations overlap.
    offsets = struct( ...
        'subreddit_probe', [  9,   3], ...
        'tfidf_lr',        [  9,   0], ...
        'text',            [-89, -14], ...
        'image',           [-62,  14], ...
        'concat',          [-10,  30], ...
        'cgf',             [  9,  -8], ...
        'llm_emb',         [-70, -13], ...
        'llm_zeroshot',    [  9,   0], ...
        'cgf_llm',         [-34,  14]);

    f  = mmid_style('figure', 6.4, 3.6);
    ax = axes('Parent', f);
    hold(ax, 'on');
    set(ax, 'XScale', 'log');

    xlim(ax, [3e-3, 5e3]);
    ylim(ax, [0.25, 0.75]);

    scatter(ax, S.cost_ms, S.macro_f1, 34, col.accent, 'filled');
    for i = 1:numel(S.cost_ms)
        if isfield(offsets, S.model{i})
            d = offsets.(S.model{i});
        else
            d = [6, 4];
        end
        mmid_offset_text(ax, S.cost_ms(i), S.macro_f1(i), d, S.label{i});
    end
    xlabel(ax, 'inference cost (ms per item, 2-core CPU)');
    ylabel(ax, 'macro-F1 (source-disjoint)');
    title(ax, 'Accuracy against inference cost', 'FontSize', 9);
    mmid_style('axes', ax);

    mmid_style('save', f, 'fig8_cost.png');
end


function mmid_offset_text(ax, x, y, offsetPoints, str)
%MMID_OFFSET_TEXT  Place text a fixed number of points from a data point.
%
%   Annotations are positioned in point offsets rather than data units so that
%   the layout survives the logarithmic horizontal axis, on which a fixed data
%   offset would appear wildly different at either end of the range.
    units = get(ax, 'Units');
    set(ax, 'Units', 'points');
    pos = get(ax, 'Position');
    set(ax, 'Units', units);

    xl = log10(xlim(ax));
    yl = ylim(ax);
    xn = (log10(x) - xl(1)) / diff(xl) + offsetPoints(1) / pos(3);
    yn = (y - yl(1)) / diff(yl) + offsetPoints(2) / pos(4);

    text(10 ^ (xl(1) + xn * diff(xl)), yl(1) + yn * diff(yl), str, ...
         'Parent', ax, 'FontSize', 6.5, 'VerticalAlignment', 'middle', ...
         'Interpreter', 'none');
end
