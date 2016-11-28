function [folderList] = getAllFolderNames( dirName, uptoLevel, currentLevel )
%GETALLFOLDERNAMES Summary of this function goes here
%   Detailed explanation goes here

    if(~exist('uptoLevel', 'var'))
       uptoLevel = 1; 
    end
    
    if(~exist('currentLevel', 'var'))
       currentLevel = 1; 
    end    
    
    if(uptoLevel < currentLevel)
       folderList = []; 
       return;
    end
    
    dirData = dir(dirName);      %# Get the data for the current directory
    dirIndex = [dirData.isdir];  %# Find the index for directories
    folderList = {dirData(dirIndex).name}';  %'# Get a list of the files
    
    if ~isempty(folderList)
        folderList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       folderList,'UniformOutput',false);
    end
    
    subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
    validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
    folderList = folderList(validIndex);
                                               %#   that are not '.' or '..'
    for iDir = find(validIndex)                 %# Loop over valid subdirectories
        nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
        folderList = [folderList; getAllFolderNames(nextDir, uptoLevel, currentLevel + 1)];  %# Recursively call getAllFiles
    end

end

