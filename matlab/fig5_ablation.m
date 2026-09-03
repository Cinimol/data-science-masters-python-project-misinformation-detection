function fig5_ablation()
%FIG5_ABLATION  Figure 5: component ablation of the proposed CGF model.
%
%   Each bar is the change in macro-F1 when one component is removed from the
%   full Consistency-Gated Fusion model, measured separately under all three
%   protocols. The figure carries an uncomfortable finding honestly: the same
%   component that helps under the random split hurts under the source-disjoint
%   split, which is precisely the behaviour expected of a component that is
%   partly exploiting source identity.
%
%   Input : data/fig5_ablation.csv
%   Output: figures_matlab/fig5_ablation.png

    S = mmid_readcsv(fullfile(mmid_style('datadir'), 'fig5_ablation.csv'));
    col = mmid_style('colours');

    variants = {'cgf', 'cgf_no_gate', 'cgf_no_cons', 'cgf_no_inter', ...
                'cgf_no_meta'};
    ticks    = {'full CGF', 'no gate', 'no consistency', 'no interaction', ...
                'no metadata'};
    protocols = {'random', 'source', 'temporal'};
    nv = numel(variants);
    np = numel(protocols);
    w  = 0.8 / np;

    f  = mmid_style('figure', 6.2, 3.4);
    ax = axes('Parent', f);
    hold(ax, 'on');

    bars = mmid_style('handles', np);
    for p = 1:np
        v = nan(nv, 1);
        for m = 1:nv
            k = find(strcmp(S.variant, variants{m}) ...
                     & strcmp(S.protocol, protocols{p}), 1);
            if ~isempty(k)
                v(m) = S.delta_macro_f1(k);
            end
        end
        bars(p) = bar(ax, (1:nv)' + (p - 1) * w, v, w);
        set(bars(p), 'FaceColor', col.(protocols{p}), 'EdgeColor', 'none');
    end

    line(ax, [0.5, nv + 1], [0 0], 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8);

    set(ax, 'XTick', (1:nv)' + w, 'XTickLabel', ticks, 'FontSize', 8);
    xlim(ax, [0.5, nv + 1]);
    ylabel(ax, 'change in macro-F1 vs full CGF', 'FontSize', 9);
    title(ax, 'Component ablation', 'FontSize', 9);
    mmid_style('legend', bars, ...
               {'random split', 'source-disjoint split', 'temporal split'}, ...
               'FontSize', 7, 'Location', 'northoutside', ...
               'Orientation', 'horizontal');
    mmid_style('axes', ax);
    set(ax, 'FontSize', 8);

    mmid_style('save', f, 'fig5_ablation.png');
end
