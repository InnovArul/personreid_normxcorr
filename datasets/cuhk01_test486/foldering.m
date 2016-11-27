% organiyes the 'same' identity images in a single folder

source = 'campus';
destination = strcat('train');

files = getAllFiles(source);

for index = 1 : size(files, 1)
   % get the image file name (ex., 000_045) & prefix (000)
   currentFilePath = files{index};
   [~, filename, extn] = fileparts(currentFilePath);
   result = filename(1:length(filename) - 1);
   
   resultFolder = strcat(destination, '/', result);
   
   %create a folder with name as prefix
   if(~exist(resultFolder, 'file'))
      mkdir(resultFolder); 
   end
   
   % save the image with name source+image file name
   img = imread(currentFilePath);
   imwrite(img, strcat(resultFolder, '/', source, '_', filename, extn));
end
