function out = mmid_style(action, varargin)
%MMID_STYLE  Shared colours, axis styling and figure export for the report.
%
%   C = MMID_STYLE('colours') returns the palette struct used throughout.
%   F = MMID_STYLE('figure', W, H) opens an invisible figure W by H inches.
%   MMID_STYLE('axes', AX) applies the common axis treatment to AX.
%   MMID_STYLE('save', F, NAME) writes F to figures_matlab/NAME.png at 160 dpi.
%   D = MMID_STYLE('datadir') returns the directory holding the exported CSVs.
%
%   Centralising these choices keeps the eight figure scripts visually
%   consistent and matches the styling of the Python originals in
%   src/analyse.py, so the two sets of figures are directly comparable.

    switch lower(action)
        case 'colours'
            out = local_colours();
        case 'figure'
            out = local_figure(varargin{:});
        case 'axes'
            local_axes(varargin{1});
            out = varargin{1};
        case 'save'
            out = local_save(varargin{:});
        case 'handles'
            out = local_handles(varargin{1});
        case 'legend'
            out = local_legend(varargin{:});
        case 'ticklabels'
            local_ticklabels(varargin{1}, varargin{2}, varargin{3});
            out = varargin{1};
        case 'alpha'
            local_alpha(varargin{1}, varargin{2});
            out = varargin{1};
        case 'datadir'
            out = fullfile(fileparts(mfilename('fullpath')), 'data');
        case 'outdir'
            out = local_outdir();
        otherwise
            error('mmid:style:action', 'Unknown action "%s".', action);
    end
end


function c = local_colours()
    c.random   = [76 114 176] / 255;    % protocol A
    c.source   = [221 132 82] / 255;    % protocol B
    c.temporal = [85 168 104] / 255;    % protocol C
    c.true     = [76 114 176] / 255;
    c.fake     = [196 78 82] / 255;
    c.accent   = [76 114 176] / 255;
    c.grey     = [0.45 0.45 0.45];
end


function f = local_figure(w, h)
    f = figure('Visible', 'off', 'Color', 'w', ...
               'Units', 'inches', 'Position', [1 1 w h], ...
               'PaperUnits', 'inches', 'PaperSize', [w h], ...
               'PaperPosition', [0 0 w h]);
end


function local_axes(ax)
    set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9, ...
            'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', ':', ...
            'GridAlpha', 0.25, 'Layer', 'top', 'LineWidth', 0.6, ...
            'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2]);
    % Subreddit names contain underscores, which TeX would silently render as
    % subscripts. Not every release exposes this property, hence the guard.
    try %#ok<TRYNC>
        set(ax, 'TickLabelInterpreter', 'none');
    end
end


function d = local_outdir()
    d = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                 'figures_matlab');
    if ~exist(d, 'dir')
        mkdir(d);
    end
end


function h = local_handles(n)
%LOCAL_HANDLES  Preallocate an array that can hold graphics handles.
%
%   MATLAB R2014b and later return graphics objects rather than numeric
%   handles, so assigning one into a preallocated ZEROS array raises
%   "value of type Bar is not convertible to double". GOBJECTS allocates the
%   right kind of array; Octave has no GOBJECTS and uses numeric handles, so
%   ZEROS is correct there.
    if exist('gobjects', 'builtin') == 5 || exist('gobjects', 'file') == 2
        h = gobjects(1, n);
    else
        h = zeros(1, n);
    end
end


function lg = local_legend(handles, labels, varargin)
%LOCAL_LEGEND  Create a legend, then style it by property assignment.
%
%   Passing Box, FontSize and Orientation straight to LEGEND is accepted by
%   some releases and rejected by others. Setting them on the returned object
%   works everywhere, and each assignment is guarded so an unsupported
%   property degrades the appearance rather than aborting the figure.
    lg = legend(handles, labels);
    for i = 1:2:numel(varargin) - 1
        try %#ok<TRYNC>
            set(lg, varargin{i}, varargin{i + 1});
        end
    end
    try %#ok<TRYNC>
        set(lg, 'Box', 'off');
    end
end


function local_ticklabels(ax, which, labels)
%LOCAL_TICKLABELS  Set tick labels without TeX mangling underscores.
%
%   Subreddit names such as "confusing_perspective" contain underscores, which
%   a TeX interpreter would render as subscripts. Disabling the interpreter
%   keeps the name exactly as it appears in the corpus.
%
%   Octave's gnuplot backend accepts the property and then ignores it, so the
%   underscore would still be drawn as a subscript there. On that backend the
%   label text is escaped instead, which gnuplot renders as a literal
%   underscore. MATLAB honours the property, and escaping it there would draw
%   the backslashes, so the escape is applied only where it is needed.
    try %#ok<TRYNC>
        set(ax, 'TickLabelInterpreter', 'none');
    end
    if iscell(labels) && local_needs_escape()
        labels = cellfun(@(s) strrep(s, '_', '\\_'), labels, ...
                         'UniformOutput', false);
    end
    set(ax, [which 'TickLabel'], labels);
end


function tf = local_needs_escape()
%LOCAL_NEEDS_ESCAPE  True on backends that ignore TickLabelInterpreter.
    tf = false;
    if ~exist('OCTAVE_VERSION', 'builtin')
        return
    end
    try %#ok<TRYNC>
        tf = strcmpi(graphics_toolkit(), 'gnuplot');
    end
end


function local_alpha(handles, value)
%LOCAL_ALPHA  Apply face transparency where the renderer supports it.
%   MATLAB honours FaceAlpha on bars, patches and scatter groups; Octave's
%   gnuplot backend does not, so a failure here is cosmetic and ignored.
    for k = 1:numel(handles)
        try %#ok<TRYNC>
            set(handles(k), 'FaceAlpha', value);
        end
    end
end


function path = local_save(f, name)
    path = fullfile(local_outdir(), name);
    print(f, '-dpng', '-r160', path);
    close(f);
    fprintf('[matlab] wrote %s\n', path);
end
