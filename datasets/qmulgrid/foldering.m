% organiyes the 'same' identity images in a single folder

source = 'train'; % gallery
sourceInternelFolders = {'gallery', 'probe'}
destination = strcat('train');


for i = 1 : length(sourceInternelFolders)
    sourcepath = strcat(source, '/', sourceInternelFolders{i});
    currentInternelFolder = sourceInternelFolders{i};
    files = getAllFiles(sourcepath);

    for index = 1 : size(files, 1)
       % get the image file name (ex., 000_045) & prefix (000)
       currentFilePath = files{index};
       [~, filename, extn] = fileparts(currentFilePath);
       result = regexp(filename, '_', 'split');
       
       resultFolder = strcat(destination, '/', result{1});
       
       %create a folder with name as prefix
       if(~exist(resultFolder, 'file'))
          mkdir(resultFolder); 
       end
       
       % save the image with name source+image file name
       %img = imread(currentFilePath);
       movefile(currentFilePath, strcat(resultFolder, '/', currentInternelFolder, '_', filename, extn));
    end
    
    % remove the source folder 
    rmdir(sourcepath, 's');
end

%------------------------------------------------
% after creation of train identity folders, create an additionalgallery folder 
% and move identities from 'train/0000' folder

additionalgalleryFolder = 'additionalgallery';
mkdir(additionalgalleryFolder)
movefile(strcat(destination, '/0000'), additionalgalleryFolder, 'f');
