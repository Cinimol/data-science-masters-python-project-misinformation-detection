function fig1_corpus_confound()
%FIG1_CORPUS_CONFOUND  Figure 1: the source/label confound in Fakeddit.
%
%   Draws one horizontal bar per source community, split by binary veracity
%   label. Because Fakeddit assigns labels by distant supervision on the
%   subreddit of origin, every bar is entirely one colour: a community is
%   wholly "true" or wholly "fake" and never mixed. That degeneracy is what
%   makes a random split recoverable from the source identity alone, and it is
%   the observation the whole project rests on.
%
%   Input : data/fig1_subreddit_counts.csv
%   Output: figures_matlab/fig1_corpus_confound.png

    S = mmid_readcsv(fullfile(mmid_style('datadir'), ...
                              'fig1_subreddit_counts.csv'));
    col = mmid_style('colours');

    n = numel(S.subreddit);
    y = (1:n)';

    f  = mmid_style('figure', 6.4, 3.6);
    ax = axes('Parent', f);
    hold(ax, 'on');

    h = barh(ax, y, [S.n_true, S.n_fake], 'stacked');
    set(h, 'BarWidth', 0.75);
    set(h(1), 'FaceColor', col.true, 'EdgeColor', 'none');
    set(h(2), 'FaceColor', col.fake, 'EdgeColor', 'none');

    set(ax, 'YTick', y, 'YDir', 'reverse');
    mmid_style('ticklabels', ax, 'Y', S.subreddit);
    ylim(ax, [0.4, n + 0.6]);
    xlabel(ax, 'posts in corpus');
    title(ax, 'Every Fakeddit subreddit maps to exactly one label');
    mmid_style('legend', h, {'labelled true', 'labelled fake'}, ...
               'Location', 'southeast');
    mmid_style('axes', ax);
    set(ax, 'FontSize', 8);

    mmid_style('save', f, 'fig1_corpus_confound.png');
end
