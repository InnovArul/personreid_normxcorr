% get all the folder names from the folder 'foldering'
source = './train';
destination = './test';

%create a folder with name 
if(~exist(destination, 'file'))
  mkdir(destination); 
end

folderList = getAllFolderNames(source);

% get the count of folders
totalFolders = size(folderList, 1);

%get random indices for train folder files
folderIndices = randsample(totalFolders, totalFolders/2);

% move the appropriate folders to 'train' folder
for folderIndex = 1:numel(folderIndices)
   from = folderList{folderIndices(folderIndex)};
   movefile(from, destination);
end
