% a MATLAB script to extract the images from cuhk03 dataset 

rootFolder = './rawdata';
dataFiles = {'labeled.mat'}; %''testsets.mat', ', 

for dataFileIndex = 1 : size(dataFiles, 1)
    
    % create a directory with the filename only
    [~, filename, extn] = fileparts(dataFiles{dataFileIndex});
    subFolder = strcat(rootFolder, '/', filename);
    mkdir(subFolder);
    
    %read the file
    data = load(dataFiles{dataFileIndex});
    internalData = getfield(data,filename);
    
    if(strcmp(filename,'testsets'))
        % copy all the mentioned data files into testsets folder
        folderNames = containers.Map;
        for dataIndex = 1:size(internalData, 1)
            currTestset = internalData{dataIndex};
                
            for testIndex = 1 : size(currTestset, 1)
                currTestInstance = currTestset(testIndex, :);
                name = strcat('campair', num2str(currTestInstance(1, 1)), '_id', num2str(currTestInstance(1, 2)));
                
                if(isKey(folderNames, name))
                   folderNames(name) =  folderNames(name) + 1;
                else
                    folderNames(name) =  1;
                end
                
            end
        end
        
  
        %take top 350 high frequency testset and copy to testsets
        %folder
        keySet = keys(folderNames);
        valuesSet = values(folderNames);

        valuesCollection = [];
        for value = valuesSet
            valuesCollection = [valuesCollection; value{:}];
        end

        [~, order] = sort(valuesCollection, 1, 'descend');
        keySet = keySet(order);

        for testset = 1 : 350
           %move folder from 'detected' to 'detected_testsets' 
           sourceFolder = strcat(rootFolder, '/detected/', keySet{testset});
           destFolder = strcat(rootFolder, '/detected_testsets/');
           movefile(sourceFolder, destFolder, 'f');

           %move folder from 'labeled' to 'labeled_testsets'
           sourceFolder = strcat(rootFolder, '/labeled/', keySet{testset});
           destFolder = strcat(rootFolder, '/labeled_testsets/');
           movefile(sourceFolder, destFolder, 'f');
        end

    else
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
        
        disp(strcat(num2str(totalFiles), ' files found in ', filename))    
    end
end
