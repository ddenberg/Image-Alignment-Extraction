function ids = get_frame_ids(names)

ids = nan(length(names), 1);

for ii = 1:length(names)

    % Find digits immediately before the file extension
    tokens = regexp(names{ii}, '_(\d+)\.[^.]+$', 'tokens');

    % Extract the frame number if one exists
    if ~isempty(tokens)
        ids(ii) = str2double(tokens{1}{1});
    end

end

end