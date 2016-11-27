%% split the folders from the folder 'train'
% into 486 identities of test

% get the names of folder
rootsource = 'train';
dirFolders = dir(rootsource);
foldernames = extractfield(dirFolders, 'name');

% eliminate dotted folder names . ..
foldernames = foldernames(strcmp(foldernames, '.') == false);
foldernames = foldernames(strcmp(foldernames, '..') == false);
foldernames = foldernames';

% randomly order the folder names
randomorder = randperm(size(foldernames, 1));
foldernames = foldernames(randomorder, :);

testsetfolders = foldernames(1:486, :);

% take first 486 folder names and copy it to testsets
dirname = 'test';

%create a folder with name, if not exists
if(~exist(dirname, 'file'))
  mkdir(dirname); 
end
    
for testsetIndex = 1:length(testsetfolders)
   source = strcat(rootsource, '/', testsetfolders(testsetIndex));
   disp(source)
   movefile(source{:}, dirname, 'f'); 
end
