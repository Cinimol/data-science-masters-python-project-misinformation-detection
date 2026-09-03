function fig2_main_results()
%FIG2_MAIN_RESULTS  Figure 2: macro-F1 for every system under all protocols.
%
%   Each model contributes three bars, one per evaluation protocol, with 95%
%   percentile-bootstrap intervals drawn as horizontal whiskers. The gap
%   between the random bars and the source-disjoint bars is the central
%   quantitative result of the project: performance that looks strong under
%   the conventional split collapses once source communities are held out.
%
%   Error bars are drawn with LINE rather than ERRORBAR because the horizontal
%   form of ERRORBAR is not portable across releases.
%
%   Input : data/fig2_main_results.csv
%   Output: figures_matlab/fig2_main_results.png

    S = mmid_readcsv(fullfile(mmid_style('datadir'), ...
                              'fig2_main_results.csv'));
    col = mmid_style('colours');

    protocols = {'random', 'source', 'temporal'};
    [models, labels] = mmid_unique_models(S);
    nm = numel(models);
    np = numel(protocols);
    h  = 0.8 / np;

    f  = mmid_style('figure', 7.2, 5.8);
    ax = axes('Parent', f);
    hold(ax, 'on');

    bars = mmid_style('handles', np);
    for p = 1:np
        v  = nan(nm, 1);
        lo = nan(nm, 1);
        hi = nan(nm, 1);
        for m = 1:nm
            k = find(strcmp(S.model, models{m}) ...
                     & strcmp(S.protocol, protocols{p}), 1);
            if ~isempty(k)
                v(m)  = S.macro_f1(k);
                lo(m) = S.lo(k);
                hi(m) = S.hi(k);
            end
        end
        ypos = (1:nm)' + (p - 1) * h;
        bars(p) = barh(ax, ypos, v, h);
        set(bars(p), 'FaceColor', col.(protocols{p}), 'EdgeColor', 'none');
        for m = 1:nm
            if ~isnan(lo(m))
                line(ax, [lo(m) hi(m)], [ypos(m) ypos(m)], ...
                     'Color', [0.3 0.3 0.3], 'LineWidth', 0.7);
            end
        end
    end

    line(ax, [0.5 0.5], [0 nm + 1], 'Color', [0.5 0.5 0.5], ...
         'LineStyle', '--', 'LineWidth', 0.8);

    set(ax, 'YTick', (1:nm)' + h * (np - 1) / 2, 'YTickLabel', labels, ...
            'YDir', 'reverse', 'FontSize', 7.5);
    ylim(ax, [0.4, nm + 1]);
    xlim(ax, [0.3, 1.0]);
    xlabel(ax, 'macro-F1 (95% bootstrap CI)', 'FontSize', 9);
    title(ax, 'Detection performance collapses once sources are held out', ...
          'FontSize', 9);
    mmid_style('legend', bars, ...
               {'random split', 'source-disjoint split', 'temporal split'}, ...
               'Location', 'northoutside', 'Orientation', 'horizontal', ...
               'FontSize', 7.5);
    mmid_style('axes', ax);
    set(ax, 'FontSize', 7.5);

    mmid_style('save', f, 'fig2_main_results.png');
end


function [models, labels] = mmid_unique_models(S)
%MMID_UNIQUE_MODELS  Model identifiers in the report's presentation order.
    [~, first] = unique(S.model, 'first');
    first  = sort(first);
    models = S.model(first);
    labels = S.label(first);
    [~, ix] = sort(S.order(first));
    models = models(ix);
    labels = labels(ix);
end

