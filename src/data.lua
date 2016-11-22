--[[
   data.lua
   
   Copyright 2015 Arulkumar <arul.csecit@ymail.com>
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301, USA.
   
   
]]--

--[[-----------------------------------------------------------

preparation of data for training and testing 

data can be different types of datasets (ethz, cuhk03, viper)

---]]-------------------------------------------------------------

require 'lfs'
require 'utilities'
require 'io'
require 'torch'
require 'pl'

logger = require 'log'
logger.outfile = opt.logFile

--data buffer to be filled
trainData = {}
testData = {}
additionalGallery = {}

if(opt.dataset == 'ethz') then
-------------------------------------------------------------------
	--load ethz dataset
	--get all file names
	trainfilepaths = {opt.datapath..'seq1', opt.datapath..'seq3'};
	testfilepaths = {opt.datapath..'seq2'};
	
	trainfiles = {};

	for index, currentFolder in ipairs(trainfilepaths) do		
		logger.trace('\tcollecting all folder names from '.. currentFolder)
		
		--now we are getting the folder names and paths inside ../seqZ
		fileNames, filePaths = getAllFileNamesInDir(currentFolder);
		logger.trace('\tNumber of files/folders : '.. #filePaths)
		
		for i, foldername in ipairs(fileNames) do
			logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
			hashKey = index .. '-' .. foldername;
			--print(hashKey)
			allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
			
			imgTable = {};
			for j, imgname in ipairs(allImgPaths) do 
			   img = loadImageWithScale(imgname)
			   imgTable[imgname] = localizeMemory(img); 
			end
			logger.trace('\tNumber of images : '.. #allImgPaths)
			trainfiles[hashKey] = imgTable;
		end
	
	end
	
	trainData = {
		data = trainfiles,
		size = function() return table.map_length(trainfiles) end
	}

	testfiles = {};

	for index, currentFolder in ipairs(testfilepaths) do		
		logger.trace('\tcollecting all folder names from '.. currentFolder)
		
		--now we are getting the folder names and paths inside ../seqZ
		fileNames, filePaths = getAllFileNamesInDir(currentFolder);
		logger.trace('\tNumber of files/folders : '.. #filePaths)
		
		for i, foldername in ipairs(fileNames) do
			logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
			hashKey = index .. '-' .. foldername;
			--logger.trace(hashKey)
			allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
			
			
			--for j, imgname in ipairs(allImgPaths) do print(imgname) end
			imgTable = {};
			for j, imgname in ipairs(allImgPaths) do 
			   img = loadImageWithScale(imgname)
			   imgTable[imgname] = localizeMemory(img); 
			end
			logger.trace('\tNumber of images : '.. #allImgPaths)
			testfiles[hashKey] = imgTable;

		end
	end	
	
	testData = {
		data = testfiles,
		size = function() return table.map_length(testfiles) end
	}
	
	logger.trace('number of train folders, test folders are ' .. table.map_length(trainfiles) .. ' / ' ..table.map_length(testfiles))
	
	--io.read()
-------------------------------------------------------------------
elseif(opt.dataset == 'cuhk03') then
	--load cuhk03 dataset
	--get all file names
  
	trainfilepaths = {opt.datapath..opt.dataType};
	
    
	trainfiles = {};
	testfiles = {};
    
    trainFilePaths = {}
    trainFileNames = {}
    validationFilePaths = {}
    validationFileNames = {}
    
    trainConfigFile = paths.concat(opt.save,'train.config')
    
    if(path.exists(trainConfigFile) == false) then 
        logger.trace('previous configuration (train.config) for training NOT found in ' .. opt.save)
        
        for index, currentFolder in ipairs(trainfilepaths) do		
            logger.trace('\tcollecting all folder names from '.. currentFolder)
            
            --now we are getting the folder names and paths inside ../seqZ
            fileNames, filePaths = getAllFileNamesInDir(currentFolder);
            logger.trace('\tNumber of folders : '.. #filePaths)
            
            --do random splitting of train set and validation set
            --print(fileNames)
            randomOrder = torch.LongTensor(table.map_length(fileNames)); 
            randomOrder:copy(torch.randperm(table.map_length(fileNames)));
            
            --collect train folder & validation folder paths separately
            trainFilePaths = table.getValues(filePaths, randomOrder[{{1, 1160}}])
            trainFileNames = table.getValues(fileNames, randomOrder[{{1, 1160}}])
                
            validationFilePaths = table.getValues(filePaths, randomOrder[{{1161, 1260}}])
            validationFileNames = table.getValues(fileNames, randomOrder[{{1161, 1260}}])
            
            --save the configuration
            config = {}
            config['trainFilePaths'] = trainFilePaths;
            config['trainFileNames'] = trainFileNames;
            config['validationFilePaths'] = validationFilePaths;
            config['validationFileNames'] = validationFileNames;
            torch.save(trainConfigFile, config, 'ascii')
        end
        
    else
        -- load the configuration
        logger.trace('previous configuration (train.config) for training found in ' .. opt.save)
        config = torch.load(trainConfigFile, 'ascii')
        trainFilePaths = config['trainFilePaths'];
        trainFileNames = config['trainFileNames'];
        validationFilePaths = config['validationFilePaths'];
        validationFileNames = config['validationFileNames'];
    end
    --io.read()
    
    -- read train files only if the current execution is not in testmode
    if(opt.testmode ~= 'test') then            
		trainfiles = loadAllImagesFromFolders(trainFileNames, trainFilePaths)
    
		trainData = {
			data = trainfiles,
			size = function() return table.map_length(trainfiles) end
		}		
	else
		logger.trace('The current execution is in TEST mode (' .. opt.testmode .. ') . So loading of train files is skipped')
	end
   
    -- during training in each epoch, the validation files are used to validate the result
    testfiles = loadAllImagesFromFolders(validationFileNames, validationFilePaths)
	
	testData = {
		data = testfiles,
		size = function() return table.map_length(testfiles) end
	}
	
	logger.trace('number of train folders, validation folders are ' .. table.map_length(trainfiles) .. ' / ' ..table.map_length(testfiles))
    
elseif(opt.dataset == 'others') then
	--load viper data set
    
    --load all image files in two folders & format
------------------------------------------------------------------
	--get all file names
	trainfilepaths = {opt.datapath..'train'};
	testfilepaths = {opt.datapath..'test'};
    additionalGalleryPath = {opt.datapath..'additionalgallery'};
	
	trainfiles = {};

	for index, currentFolder in ipairs(trainfilepaths) do		
		logger.trace('\tcollecting all folder names from '.. currentFolder)
		
		--now we are getting the folder names and paths inside ../seqZ
		fileNames, filePaths = getAllFileNamesInDir(currentFolder);
		logger.trace('\tNumber of files/folders : '.. #filePaths)
        
        trainfiles = loadAllImagesFromFolders(fileNames, filePaths)
		
        --[[]
		for i, foldername in ipairs(fileNames) do
			logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
			hashKey = index .. '-' .. foldername;
			--print(hashKey)
			allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
			
			imgTable = {};
			for j, imgname in ipairs(allImgPaths) do 
			   img = loadImageWithScale(imgname)
			   imgTable[imgname] = (img);  --localizeMemory
			end
			logger.trace('\tNumber of images : '.. #allImgPaths)
			trainfiles[hashKey] = imgTable;
		end
        --]]
	
	end
	
	trainData = {
		data = trainfiles,
		size = function() return table.map_length(trainfiles) end
	}

	testfiles = {};

	for index, currentFolder in ipairs(testfilepaths) do		
		logger.trace('\tcollecting all folder names from '.. currentFolder)
		
		--now we are getting the folder names and paths inside ../seqZ
		fileNames, filePaths = getAllFileNamesInDir(currentFolder);
		logger.trace('\tNumber of files/folders : '.. #filePaths)
		
        testfiles = loadAllImagesFromFolders(fileNames, filePaths)
        --[[]
		for i, foldername in ipairs(fileNames) do
			logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
			hashKey = index .. '-' .. foldername;
			--print(hashKey)
			allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
			
			
			--for j, imgname in ipairs(allImgPaths) do print(imgname) end
			imgTable = {};
			for j, imgname in ipairs(allImgPaths) do 
			   img = loadImageWithScale(imgname)
			   imgTable[imgname] = (img); --localizeMemory
			end
			logger.trace('\tNumber of additional gallery images : '.. #allImgPaths)
			testfiles[hashKey] = imgTable;

		end
        --]]
	end	
	
	testData = {
		data = testfiles,
		size = function() return table.map_length(testfiles) end
	}
	
	additionalGalleryFiles = {};

	for index, currentFolder in ipairs(additionalGalleryPath) do		
		logger.trace('\tcollecting all folder names from '.. currentFolder)
		
		--now we are getting the folder names and paths inside ../seqZ
		fileNames, filePaths = getAllFileNamesInDir(currentFolder);
		logger.trace('\tNumber of files/folders : '.. #filePaths)
		
		for i, foldername in ipairs(fileNames) do
			logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
			hashKey = index .. '-' .. foldername;
			--print(hashKey)
			allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
			
			
			--for j, imgname in ipairs(allImgPaths) do print(imgname) end
			imgTable = {};
			for j, imgname in ipairs(allImgPaths) do 
			   img = loadImageWithScale(imgname)
			   imgTable[imgname] = (img); --localizeMemory
			end
			logger.trace('\tNumber of images : '.. #allImgPaths)
			additionalGalleryFiles[hashKey] = imgTable;

		end
	end	
	
	additionalGallery = {
		data = additionalGalleryFiles,
		size = function() return table.map_length(additionalGalleryFiles) end
	}
	
	logger.trace('number of train folders, test folders, additional gallery folders are ' .. table.map_length(trainfiles) .. ' / ' ..table.map_length(testfiles) .. ' / ' ..table.map_length(additionalGalleryFiles))
else

	logger.trace('unknown dataset!');

end
