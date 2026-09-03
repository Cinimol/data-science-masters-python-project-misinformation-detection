function S = mmid_readcsv(filename)
%MMID_READCSV  Read a header-row CSV into a struct of columns.
%
%   S = MMID_READCSV(FILENAME) returns a struct whose fields are named after
%   the columns of FILENAME. A column whose every non-empty entry parses as a
%   number becomes a double column vector; any other column becomes a cell
%   array of character vectors.
%
%   The reader uses FILEREAD, REGEXP and SSCANF rather than READTABLE or
%   TEXTSCAN. READTABLE is unavailable in Octave, and TEXTSCAN's format and
%   delimiter handling differs between MATLAB releases and between MATLAB and
%   Octave, which made it the least portable part of this suite. The three
%   functions used here behave identically everywhere and need no toolbox.
%
%   Files whose body is entirely numeric take a vectorised SSCANF path, which
%   matters for the two largest inputs: fig3_item_consistency.csv has 21,798
%   rows and fig7_gate_source.csv has 6,870.
%
%   See also MMID_CHECK, MMID_STYLE.

    if exist(filename, 'file') ~= 2
        [~, base, ext] = fileparts(filename);
        error('mmid:readcsv:notFound', ...
              ['Cannot find "%s%s".\n' ...
               'Expected it in: %s\n' ...
               'Generate the plotting data by running this at the ' ...
               'repository root:\n    python -m src.export_matlab\n' ...
               'Then run mmid_check from the matlab/ directory.'], ...
              base, ext, fileparts(filename));
    end

    txt = fileread(filename);
    lines = regexp(txt, '\r\n|\r|\n', 'split');
    lines = lines(~cellfun(@isempty, strtrim(lines)));
    if isempty(lines)
        error('mmid:readcsv:empty', '"%s" is empty.', filename);
    end

    names = strtrim(regexp(lines{1}, ',', 'split'));
    ncol  = numel(names);
    body  = lines(2:end);
    nrow  = numel(body);
    fields = cell(1, ncol);
    for k = 1:ncol
        fields{k} = mmid_fieldname(names{k});
    end

    S = struct();
    if nrow == 0
        for k = 1:ncol
            S.(fields{k}) = [];
        end
        return
    end

    % ---- fast path: an entirely numeric body ---------------------------- %
    joined = sprintf('%s\n', body{:});
    fmt = [repmat('%f,', 1, ncol - 1) '%f'];
    [v, count] = sscanf(joined, fmt);
    if count == ncol * nrow
        M = reshape(v, ncol, nrow).';
        for k = 1:ncol
            S.(fields{k}) = M(:, k);
        end
        return
    end

    % ---- general path: split each row, then type each column ------------ %
    cols = cell(1, ncol);
    for k = 1:ncol
        cols{k} = cell(nrow, 1);
    end
    for i = 1:nrow
        parts = mmid_splitrow(body{i}, ncol);
        for k = 1:ncol
            cols{k}{i} = parts{k};
        end
    end

    for k = 1:ncol
        raw = cols{k};
        value = str2double(raw);
        filled = ~cellfun(@isempty, strtrim(raw));
        if any(filled) && all(~isnan(value(filled)))
            S.(fields{k}) = value(:);
        else
            S.(fields{k}) = raw(:);
        end
    end
end


function parts = mmid_splitrow(line, ncol)
%MMID_SPLITROW  Split one CSV row, honouring double-quoted fields.
%
%   Padded or truncated to NCOL entries, so a short trailing row cannot throw
%   an index error part-way through a file.
    raw = regexp(line, ',', 'split');
    if any(line == '"')
        raw = mmid_splitquoted(line);
    end
    parts = cell(1, ncol);
    for k = 1:ncol
        if k <= numel(raw)
            parts{k} = strtrim(strrep(raw{k}, '"', ''));
        else
            parts{k} = '';
        end
    end
end


function out = mmid_splitquoted(line)
%MMID_SPLITQUOTED  Comma split that ignores commas inside double quotes.
    out = {};
    current = '';
    inQuote = false;
    for i = 1:numel(line)
        c = line(i);
        if c == '"'
            inQuote = ~inQuote;
        elseif c == ',' && ~inQuote
            out{end + 1} = current;                             %#ok<AGROW>
            current = '';
        else
            current(end + 1) = c;                               %#ok<AGROW>
        end
    end
    out{end + 1} = current;
end


function name = mmid_fieldname(raw)
%MMID_FIELDNAME  Coerce a CSV column name into a legal struct field name.
    name = regexprep(raw, '[^A-Za-z0-9_]', '_');
    if isempty(name) || ~isletter(name(1))
        name = ['x_' name];
    end
end
