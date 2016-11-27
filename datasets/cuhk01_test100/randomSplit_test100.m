%% split the folders from the folder 'labeled' 'detected'
% into 100 identities of labeled_testsets & detected_testsets

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

testsetfolders = foldernames(1:100, :);
 
% take first 100 folder names and copy it to testsets
dirname = 'test';
%create a folder with name as prefix
if(~exist(dirname, 'file'))
  mkdir(dirname); 
end

for testsetIndex = 1:length(testsetfolders)
   source = strcat(rootsource, '/', testsetfolders(testsetIndex));
   disp(strcat(source, {' --> '}, dirname))
   movefile(source{:}, dirname, 'f'); 
end
