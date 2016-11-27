% a MATLAB script to extract the images from cuhk03 dataset 

rootFolder = './rawdata';

data = load('cuhk-03.mat');

dataLabels = {'labeled', 'detected'};

for dataLabelsIndex = 1 : length(dataLabels)
    
    % create a directory with the label only
    label = dataLabels{dataLabelsIndex};
    subFolder = strcat(rootFolder, '/', label);
    mkdir(subFolder);
    
    %read the file
    internalData = getfield(data,label);

    totalFiles = 0;  
    
    %for each pair of cameras, go through all the identities and save all the pictures in a folder
    for pairIndex = 1:size(internalData, 1)
      
        % do not consider the camera pairs 4 & 5 to get 1360 identities
        if(pairIndex > 3)
            break
        end
        
        pairFolderName = strcat(subFolder, '/', 'campair', num2str(pairIndex), '_');

        imgData = internalData{pairIndex};

        % for each identity, store the available 10 (or less) images
        for identity = 1 : size(imgData, 1)
            % create a folder for this identity
            identityFolderName = strcat(pairFolderName, 'id', num2str(identity));
            mkdir(identityFolderName);

            %collect 10 images
            for imgIndex = 1 : 10
                imgName = strcat(identityFolderName, '/', num2str(imgIndex), '.png');
                img = imgData{identity, imgIndex};

                if(size(img, 1) ~= 0)
                    imwrite(img, imgName);
                    totalFiles = totalFiles + 1;
                end
            end
        end
    end    
    
    disp(strcat(num2str(totalFiles), {' files found in '}, label))    
end
