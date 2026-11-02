function ids = get_frame_ids(names)

ids = zeros(length(names), 1);

for ii = 1:length(names)

    % Find all groups of consecutive digits in the filename
    tokens = regexp(names{ii}, '\d+', 'match');

    % Extract the last group of digits as the frame number
    ids(ii) = str2double(tokens{end});

end

end