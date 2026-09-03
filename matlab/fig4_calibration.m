function fig4_calibration(protocol)
%FIG4_CALIBRATION  Figure 4: reliability diagrams for the leading detectors.
%
%   FIG4_CALIBRATION(PROTOCOL) plots, for each of three detectors, the mean
%   predicted probability of falsehood in each of ten bins against the observed
%   frequency of falsehood in that bin. Points on the diagonal indicate a
%   probability that can be taken at face value. Calibration matters here
%   because a moderation pipeline thresholds these scores, so a detector that
%   is confidently wrong is more damaging than one that is merely inaccurate.
%
%   Called with no argument, every available protocol is drawn in turn.
%
%   Input : data/fig4_calibration_<protocol>.csv
%   Output: figures_matlab/fig4_calibration_<protocol>.png

    if nargin < 1
        for p = {'random', 'source', 'temporal'}
            fig4_calibration(p{1});
        end
        return
    end

    file = fullfile(mmid_style('datadir'), ...
                    sprintf('fig4_calibration_%s.csv', protocol));
    if ~exist(file, 'file')
        fprintf('[matlab] skipping fig4 (%s): no exported data\n', protocol);
        return
    end
    S = mmid_readcsv(file);

    f  = mmid_style('figure', 3.8, 3.6);
    ax = axes('Parent', f);
    hold(ax, 'on');

    plot(ax, [0 1], [0 1], 'k--', 'LineWidth', 0.8);

    models  = unique_stable(S.model);
    markers = {'o', 's', '^', 'd', 'v'};
    palette = [ 76 114 176; 196  78  82; 221 132  82; 85 168 104] / 255;
    handles = mmid_style('handles', numel(models));
    names   = cell(1, numel(models));
    for m = 1:numel(models)
        k = strcmp(S.model, models{m});
        [x, ix] = sort(S.mean_pred(k));
        obs = S.observed(k);
        handles(m) = plot(ax, x, obs(ix), ...
                          'Marker', markers{mod(m - 1, numel(markers)) + 1}, ...
                          'MarkerSize', 3.5, 'LineWidth', 1.2, ...
                          'Color', palette(mod(m - 1, size(palette, 1)) + 1, :), ...
                          'MarkerFaceColor', ...
                          palette(mod(m - 1, size(palette, 1)) + 1, :));
        idx = find(k, 1);
        names{m} = S.label{idx};
    end

    xlim(ax, [0 1]);
    ylim(ax, [0 1]);
    xlabel(ax, 'predicted P(fake)');
    ylabel(ax, 'observed frequency');
    title(ax, sprintf('Calibration (%s split)', protocol), 'FontSize', 9);
    mmid_style('legend', [plot_dummy(ax), handles], ...
               [{'perfect calibration'}, names], ...
               'Location', 'southeast', 'FontSize', 6.5);
    mmid_style('axes', ax);

    mmid_style('save', f, sprintf('fig4_calibration_%s.png', protocol));
end


function h = plot_dummy(ax)
%PLOT_DUMMY  A hidden line standing in for the diagonal in the legend.
    h = plot(ax, NaN, NaN, 'k--', 'LineWidth', 0.8);
end


function u = unique_stable(c)
%UNIQUE_STABLE  Unique entries of a cellstr in order of first appearance.
    [~, first] = unique(c, 'first');
    u = c(sort(first));
end
