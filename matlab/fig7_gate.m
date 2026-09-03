function r = fig7_gate(protocol)
%FIG7_GATE  Figure 7: does the learned gate track cross-modal consistency?
%
%   R = FIG7_GATE(PROTOCOL) plots each held-out item as a point, with CLIP
%   image/headline agreement on the horizontal axis and the mean activation of
%   the model's learned visual gate on the vertical axis, and returns the
%   Pearson correlation between them. A positive correlation is evidence that
%   the gate has learned the intended behaviour, namely attenuating the visual
%   pathway when the image and headline disagree, rather than becoming an
%   arbitrary learned constant.
%
%   Called with no argument, every available protocol is drawn in turn.
%
%   Input : data/fig7_gate_<protocol>.csv
%   Output: figures_matlab/fig7_gate_<protocol>.png

    if nargin < 1
        for p = {'random', 'source', 'temporal'}
            fig7_gate(p{1});
        end
        r = [];
        return
    end

    file = fullfile(mmid_style('datadir'), ...
                    sprintf('fig7_gate_%s.csv', protocol));
    if ~exist(file, 'file')
        fprintf('[matlab] skipping fig7 (%s): no exported data\n', protocol);
        r = [];
        return
    end
    S = mmid_readcsv(file);
    col = mmid_style('colours');

    c = corrcoef(S.gate, S.clip_sim);
    r = c(1, 2);

    f  = mmid_style('figure', 4.0, 3.4);
    ax = axes('Parent', f);
    hold(ax, 'on');

    classes = {0, 'true', col.true; 1, 'fake', col.fake};
    handles = mmid_style('handles', 2);
    keys    = mmid_style('handles', 2);
    for k = 1:2
        m = S.is_fake == classes{k, 1};
        handles(k) = scatter(ax, S.clip_sim(m), S.gate(m), 3, ...
                             classes{k, 3}, 'filled');
        % A hidden marker of the same colour stands in for the scatter group
        % in the legend: not every renderer builds a legend entry from a
        % scatter handle, but every renderer does so from a line handle.
        keys(k) = plot(ax, NaN, NaN, 'o', 'MarkerSize', 4, ...
                       'MarkerFaceColor', classes{k, 3}, ...
                       'MarkerEdgeColor', 'none', 'LineStyle', 'none');
    end
    mmid_style('alpha', handles, 0.25);

    xlabel(ax, 'CLIP cosine(image, headline)');
    ylabel(ax, 'mean visual gate activation');
    title(ax, sprintf('Gate vs consistency, %s split (r = %.2f)', ...
                      protocol, r), 'FontSize', 9);
    mmid_style('legend', keys, classes(:, 2), 'Location', 'northwest');
    mmid_style('axes', ax);

    mmid_style('save', f, sprintf('fig7_gate_%s.png', protocol));
end
