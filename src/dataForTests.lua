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
	
	logger.trace('number of train files, test files are ' .. table.map_length(trainfiles) .. ' / ' ..table.map_length(testfiles))
	
	--io.read()
-------------------------------------------------------------------
elseif(opt.dataset == 'cuhk03') then
	--load cuhk03 dataset
	--get all file names
	filepaths = {opt.datapath .. opt.dataType .. '_testsets'};
	
	testfiles = {};
    
    testFilePaths = {}
    testFileNames = {}
    
    for index, currentFolder in ipairs(filepaths) do		
        logger.trace('\tcollecting all folder names from '.. currentFolder)
        
        --now we are getting the folder names and paths inside ../seqZ
        testFileNames, testFilePaths = getAllFileNamesInDir(currentFolder);
        logger.trace('\tNumber of folders : '.. #testFilePaths)         
    end
   
    
    -- during training in each epoch, the validation files are used to validate the result
    testfiles = loadAllImagesFromFolders(testFileNames, testFilePaths)
    --print(testfiles)

	testData = {
		data = testfiles,
		size = function() return table.map_length(testfiles) end
	}
	
	print(table.map_length(testfiles))
	--io.read()
	
	logger.trace('number of test files are ' ..table.map_length(testfiles))
    
elseif(opt.dataset == 'others') then
	--load viper data set
    
    --load all image files in two folders & format
------------------------------------------------------------------
	--get all file names
	testfilepaths = {opt.datapath..'test'};
    additionalGalleryPath = {opt.datapath..'additionalgallery'};
	
	testfiles = {};

	for index, currentFolder in ipairs(testfilepaths) do		
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
			   imgTable[imgname] = localizeMemory(img); 
			end
			logger.trace('\tNumber of images : '.. #allImgPaths)
			additionalGalleryFiles[hashKey] = imgTable;

		end
	end	
	
	additionalGallery = {
		data = additionalGalleryFiles,
		size = function() return table.map_length(testfiles) end
	}
	
	logger.trace('number of test files, additional gallery files are ' .. table.map_length(testfiles) .. ' / ' ..table.map_length(additionalGalleryFiles))
    --io.read()
          
else

	logger.trace('unknown dataset!');

end
