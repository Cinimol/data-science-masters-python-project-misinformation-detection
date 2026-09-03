function fig6_error_by_category(protocol)
%FIG6_ERROR_BY_CATEGORY  Figure 6: which kinds of misinformation survive?
%
%   FIG6_ERROR_BY_CATEGORY(PROTOCOL) breaks the proposed detector's accuracy
%   down by the fine-grained Fakeddit category of the item, annotating each bar
%   with its support. The ordering is deliberately worst-first: the categories
%   at the top are where a deployed system would fail, and reporting them is
%   more useful than a single headline accuracy.
%
%   Called with no argument, every available protocol is drawn in turn.
%
%   Input : data/fig6_error_by_category_<protocol>.csv
%   Output: figures_matlab/fig6_error_by_category_<protocol>.png

    if nargin < 1
        for p = {'random', 'source', 'temporal'}
            fig6_error_by_category(p{1});
        end
        return
    end

    file = fullfile(mmid_style('datadir'), ...
                    sprintf('fig6_error_by_category_%s.csv', protocol));
    if ~exist(file, 'file')
        fprintf('[matlab] skipping fig6 (%s): no exported data\n', protocol);
        return
    end
    S = mmid_readcsv(file);
    col = mmid_style('colours');

    n = numel(S.accuracy);
    f  = mmid_style('figure', 5.2, 2.8);
    ax = axes('Parent', f);
    hold(ax, 'on');

    b = barh(ax, (1:n)', S.accuracy, 0.7);
    set(b, 'FaceColor', col.accent, 'EdgeColor', 'none');
    for i = 1:n
        text(S.accuracy(i) + 0.015, i, sprintf('n=%d', S.n(i)), ...
             'Parent', ax, 'VerticalAlignment', 'middle', 'FontSize', 7);
    end

    set(ax, 'YTick', 1:n, 'FontSize', 8);
    mmid_style('ticklabels', ax, 'Y', S.category);
    ylim(ax, [0.4, n + 0.6]);
    xlim(ax, [0, 1.15]);
    xlabel(ax, 'per-category accuracy', 'FontSize', 9);
    title(ax, sprintf('%s, %s split', S.model{1}, protocol), 'FontSize', 9);
    mmid_style('axes', ax);
    set(ax, 'FontSize', 8);

    mmid_style('save', f, sprintf('fig6_error_by_category_%s.png', protocol));
end
