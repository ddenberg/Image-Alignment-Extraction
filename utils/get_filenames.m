function [names, folder] = get_filenames(path, substring_include, substring_exclude)

listing = dir(fullfile(path, '**/*.*'));
names = {listing.name};
names = names(:);
folder = {listing.folder};
folder = folder(:);

filter_exclude = false(length(names), 1);
if iscell(substring_exclude)
    for ii = 1:length(substring_exclude)
        filter_exclude = filter_exclude | contains(names, substring_exclude{ii}, 'IgnoreCase', true);
    end
else
    filter_exclude = contains(names, substring_exclude, 'IgnoreCase', true);
end

filter_include = true(length(names), 1);
if iscell(substring_include)
    for ii = 1:length(substring_include)
        filter_include = filter_include & contains(names, substring_include{ii}, 'IgnoreCase', true);
    end
else
    filter_include = contains(names, substring_include, 'IgnoreCase', true);
end

names = names(filter_include & ~filter_exclude);
folder = folder(filter_include & ~filter_exclude);

end

