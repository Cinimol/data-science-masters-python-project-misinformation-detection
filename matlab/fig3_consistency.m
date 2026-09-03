function fig3_consistency()
%FIG3_CONSISTENCY  Figure 3: is CLIP image/headline agreement informative?
%
%   The left panel overlays the distribution of CLIP cosine similarity for
%   posts labelled true and posts labelled fake; the separation is real but
%   small, which is why consistency is treated as one gating input rather than
%   as a detector in its own right. The right panel averages the same quantity
%   per source community, showing that most of the between-class difference is
%   carried by community-level style rather than by individual mismatches.
%
%   Histograms are computed with HISTC and drawn with BAR so that no
%   Statistics Toolbox and no recent-release HISTOGRAM call is required.
%
%   Inputs : data/fig3_item_consistency.csv
%            data/fig3_subreddit_consistency.csv
%   Output : figures_matlab/fig3_consistency.png

    d     = mmid_style('datadir');
    items = mmid_readcsv(fullfile(d, 'fig3_item_consistency.csv'));
    comms = mmid_readcsv(fullfile(d, 'fig3_subreddit_consistency.csv'));
    col   = mmid_style('colours');

    f = mmid_style('figure', 7.6, 3.6);

    % ---- left panel: per-item distribution by veracity class ------------- %
    ax1  = axes('Parent', f, 'Position', [0.07 0.17 0.38 0.72]);
    hold(ax1, 'on');
    edges   = linspace(min(items.clip_sim), max(items.clip_sim), 61);
    centres = edges(1:end-1) + diff(edges) / 2;
    width   = edges(2) - edges(1);
    classes = {0, 'true', col.true; 1, 'fake', col.fake};
    handles = mmid_style('handles', 2);
    for k = 1:2
        v = items.clip_sim(items.is_fake == classes{k, 1});
        c = histc(v, edges);
        c = c(1:end-1);
        density = c / (sum(c) * width);
        handles(k) = bar(ax1, centres, density, 1.0);
        set(handles(k), 'FaceColor', classes{k, 3}, 'EdgeColor', 'none');
    end
    mmid_style('alpha', handles, 0.55);
    xlabel(ax1, 'CLIP cosine(image, headline)');
    ylabel(ax1, 'density');
    title(ax1, 'Consistency by veracity class', 'FontSize', 9);
    mmid_style('legend', handles, classes(:, 2), 'Location', 'northwest');
    mmid_style('axes', ax1);

    % ---- right panel: community means, coloured by that community's label - %
    ax2 = axes('Parent', f, 'Position', [0.62 0.17 0.35 0.72]);
    hold(ax2, 'on');
    n = numel(comms.mean_sim);
    trueOnly = comms.mean_sim;
    fakeOnly = comms.mean_sim;
    trueOnly(comms.is_fake == 1) = NaN;
    fakeOnly(comms.is_fake == 0) = NaN;
    bt = barh(ax2, (1:n)', trueOnly, 0.72);
    set(bt, 'FaceColor', col.true, 'EdgeColor', 'none');
    bf = barh(ax2, (1:n)', fakeOnly, 0.72);
    set(bf, 'FaceColor', col.fake, 'EdgeColor', 'none');
    set(ax2, 'YTick', 1:n, 'FontSize', 6);
    mmid_style('ticklabels', ax2, 'Y', comms.subreddit);
    ylim(ax2, [0.4, n + 0.6]);
    xlabel(ax2, 'mean CLIP cosine', 'FontSize', 9);
    title(ax2, 'Mean consistency by community', 'FontSize', 9);
    mmid_style('axes', ax2);
    set(ax2, 'FontSize', 6);

    mmid_style('save', f, 'fig3_consistency.png');
end
