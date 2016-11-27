%% split the folders from the folder 'labeled' 'detected'
% into 100 identities of labeled_testsets & detected_testsets

directories = {'labeled', 'detected'};

% get the names of folders (or) identities from any of the (labeled or detected) dataset
dirFolders = dir('detected');
foldernames = extractfield(dirFolders, 'name');

% eliminate dotted folder names . ..
foldernames = foldernames(strcmp(foldernames, '.') == false);
foldernames = foldernames(strcmp(foldernames, '..') == false);
foldernames = foldernames';

% randomly order the folder names
randomorder = randperm(size(foldernames, 1));
foldernames = foldernames(randomorder, :);

testsetfolders = foldernames(1:100, :);

% take first 100 folder names and copy it to testsets
for index = 1:length(directories)
    dirname = directories{index};
    
    % create testset folder
    testsetname = strcat(dirname, '_testsets');
    mkdir(testsetname);
    
    for testsetIndex = 1:length(testsetfolders)
       destination = strcat(dirname, '/', testsetfolders(testsetIndex));
       movefile(destination{:}, testsetname, 'f'); 
    end
    
end
