function ids = get_frame_ids(names)

ids = cell(length(names), 1);

for ii = 1:length(names)
    % [start_idx, end_idx] = regexp(names{ii}, '[0-9]\w+');
    idx = regexp(names{ii}, '[0-9]');
    start_idx = min(idx);
    end_idx = max(idx);
    str = names{ii}(start_idx:end_idx);
    str = split(str, '_');
    num = str2double(str);

    ids{ii} = num(:).';

end

ids = cell2mat(ids);

end