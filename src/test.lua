--[[
   test.lua
   
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
----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nngraph'
require 'nn'
require 'utilities'
require 'image'
logger = require 'log'
matio = require 'matio'
dofile 'rankScores.lua'

logger.outfile = opt.logFile

----------------------------------------------------------------------
logger.trace('==> defining test procedure')

numberOfgalleryImages = opt.galleryimages;
if(numberOfgalleryImages == nil) then
    numberOfgalleryImages = 1;
end

testdatanames = {}
allTestData = {}
gallaryImageData = {}
identityIndices = {}

-- test function
function test(isDisplay)
    -- define if logging trace display is required
	if(isDisplay == nil) then isDisplay = true; end
 
    -- local vars
    local time = sys.clock()
    
    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model:evaluate()
    
    --collect the gallery images only if it has not been collected already
    if(totalGalleryImages == nil or opt.testmode == 'test') then
      testDataCopied = deepcopy(testData);
      allTestData = testDataCopied['data'];
      testdatanames = table.getAllKeys(testDataCopied['data']);
      
      totalTestPersons = #testdatanames;
      
      if(opt.testDataCount) then
          totalTestPersons = opt.testDataCount
      end
      
      local inputs = {}
      local targets = {}
          
      gallaryImageData = {};
      
      -- test over test data
      --NEW PROCEDURE: (to be implemented)
      -- for each person, randomly select an image as gallery image & delete that image from that particular person
      -- now foreach probe image, pass all the gallery images and get the scores
      logger.info('collecting all the gallery images : \n')
      totalProbeImages = 0;
      totalGalleryImages = 0;
      identityIndices = {}   
      
      -- collect gallery images 
      -- here each testperson (identity) gives out a gallery image
      -- TODO: for singleshot = 1 image, multishot = 2 images (CUHK01_test486 )
      for identityIndex = 1, totalTestPersons do
          
        --get current test sequence name
          currentSequenceName = testdatanames[identityIndex];
          currentGalleryTable = {};
          identityIndices[currentSequenceName] = identityIndex; 
          
  --        if(t == 1) then
  --            print(allTestData[currentSequenceName])
  --            io.read()
  --        end

          -- get all the images for particular identity
          allImgNames = table.getAllKeys(allTestData[currentSequenceName])
          gallaryImageData[currentSequenceName] = {};
          
          --select the first sample for the current identity        
          --get random index for gakkery image
          for galleryImageIndex = 1, numberOfgalleryImages do
            --skipping images to compensate view variations
            currentGallerySeletectedIndex = getRandomNumber(1, #allImgNames);
            
            --gallery image selection for 'others' + 'qmulgrid'
            if (opt.dataset == 'others') and ((opt.datasetname == 'qmulgrid') or 
                                            (opt.datasetname == 'prid450s') or 
                                            (opt.datasetname == 'viper')) then 
                currentGallerySeletectedIndex = 1;
            end
            
            galleryImagePath = allImgNames[currentGallerySeletectedIndex]; 
            galleryImage = allTestData[currentSequenceName][galleryImagePath];
            
            -- insert the current gallery image table into Gallery table
            gallaryImageData[currentSequenceName][galleryImagePath] = galleryImage;
            
            -- delete the selected gallery image from parent table
            allTestData[currentSequenceName][galleryImagePath] = nil;
            totalGalleryImages = totalGalleryImages + 1;
          end
          
          totalProbeImages = totalProbeImages + #allImgNames - numberOfgalleryImages;
      end

      --
      --  gallaryImageData
      --       ---> person name or ID:
      --                 --> filepath1 = image
      --                 --> filepath2 = image    
      --       ---> person name or ID:
      --                 --> filepath1 = image    
      --                     ....

      ------------------------------------------------------------------------
      --------- Insert additional gallery images if available-----------------

      additionalGalleryImgCount = 0;
      if(next(additionalGallery) ~= nil) then
        allAdditionalGalleryIDs = table.getAllKeys(additionalGallery.data);
        
        for k,v in ipairs(allAdditionalGalleryIDs) do
          gallaryImageData[v] = additionalGallery.data[v];
          identityIndices[v] = 0;
          additionalGalleryImgCount = additionalGalleryImgCount + table.map_length(additionalGallery.data[v]);
        end
      end
    
    end --**** end if(totalGalleryImages == nil) then

    totalGalleryClasses = table.map_length(gallaryImageData);
    logger.trace("total gallery classes (identities (including additional gallery imgs)): " .. totalGalleryClasses);
    logger.trace('total probe classes : ' .. totalTestPersons);    
    logger.trace("total gallery images (including additional gallery imgs): " .. totalGalleryImages + additionalGalleryImgCount);
    logger.trace('total probe images (estimated) : ' .. totalProbeImages);

    --io.read()
    --do return end
    
    ---- after generating all positive, negative pairs, split them into batch size of 64
    -- then for each batch size of 64, do the stochastic gradient descent
    -- scores is a table with probeimages as rows, and gallery images as columns
    -- scores will contain the model score for the comparision of particular probe and gallery images
    scores = torch.Tensor(totalProbeImages, totalGalleryImages + additionalGalleryImgCount, 3);
    correctMatches = torch.Tensor(totalProbeImages);
    actualProbeImagesCount = 1;

    --io.read()
    -- log the probeimages
    logger.trace('Probe images: ')
     
    for probeIdIndex = 1, totalTestPersons do
        currentSequenceName = testdatanames[probeIdIndex];
        if(isDisplay) then logger.trace(currentSequenceName) end
        -- get all the images for particular identity
        allImgNames = table.getAllKeys(allTestData[currentSequenceName]) 
        for probeImgIndex = 1, #allImgNames do
			if(isDisplay) then 
				logger.trace(actualProbeImagesCount .. ': ' .. allImgNames[probeImgIndex])
            end
            actualProbeImagesCount = actualProbeImagesCount + 1
        end
    end
    logger.trace('Probe images (Actual): ' .. (actualProbeImagesCount - 1))

    -- log the probeimages
    logger.trace('Gallery images: ')
    galleryIdentities = table.getAllKeys(gallaryImageData)
    actualGalleryIndex = 1;
    -- go through all the identity in gallery (including additional gallery images)
    -- *******************************************              
    for galleryIdIndex = 1, #galleryIdentities do
      currentGalleryIdentity = gallaryImageData[galleryIdentities[galleryIdIndex]];
      allCurrentGalleryImgNames = table.getAllKeys(currentGalleryIdentity)
      
      -- for each image in a particular identity, compare it with probe image
      for localGalleryImageIndex = 1, #allCurrentGalleryImgNames do
      -- *******************************************                
        --current gallery ID
        if(isDisplay) then 
			logger.trace(identityIndices[galleryIdentities[galleryIdIndex]] .. ' : ' .. allCurrentGalleryImgNames[localGalleryImageIndex]);
		end
        actualGalleryIndex = actualGalleryIndex + 1;
      end
    end
   logger.trace('Gallery images (Actual): ' .. (actualGalleryIndex-1))  

    --buffer to hold gallery and probe image details
    probeImageDetails = {} 
    galleryImageDetails = {}   
    local totalError = 0;   

    actualProbeImagesCount = 1;
    -- probeIdIndex = identity index
    -- for each identity, go through all the available probe images and compare with gallery images
    -- **********************************************
    for probeIdIndex = 1, totalTestPersons do
        -- disp progress
        xlua.progress(probeIdIndex, totalTestPersons)
        --get current test sequence name
        currentSequenceName = testdatanames[probeIdIndex];

--
--        if(probeIdIndex == 1) then
--            print(allTestData[currentSequenceName])
--            io.read()
--            print(gallaryImageData)
--            io.read()
--        end        
--

        -- get all the images for particular identity
        allImgNames = table.getAllKeys(allTestData[currentSequenceName]) 

        -- for each image in the probe identity, compare it with all the gallery images
        -- *******************************************        
        for probeImgIndex = 1, #allImgNames do
                       
            currentProbeImage = localizeMemory(allTestData[currentSequenceName][allImgNames[probeImgIndex]]);

            -- pass all gallery images
            -- loop through all the galary images including additional gallery images of particular identity
            galleryIdentities = table.getAllKeys(gallaryImageData)
            actualGalleryIndex = 1;
            
            -- go through all the identity in gallery (including additional gallery images)
            -- *******************************************              
            for galleryIdIndex = 1, #galleryIdentities do
            
              currentGalleryIdentity = gallaryImageData[galleryIdentities[galleryIdIndex]];
              
              allCurrentGalleryImgNames = table.getAllKeys(currentGalleryIdentity)
              
              -- for each image in a particular identity, compare it with probe image
              for localGalleryImageIndex = 1, #allCurrentGalleryImgNames do
              -- *******************************************                
                galleryImagePath = allCurrentGalleryImgNames[localGalleryImageIndex];
                
                --current gallery ID
                galleryImage = localizeMemory(currentGalleryIdentity[galleryImagePath]);
                
                --print('\n\nexpected : ' .. targets[randomOrder[i]])
                --image.display(newimg1);
                --image.display(newimg2);

                -- estimate f
                local pred = localizeMemory(model:forward({galleryImage, currentProbeImage}))

                --prepare target for confusion matrix
                target = 2
                if(currentSequenceName == galleryIdentities[galleryIdIndex]) then
                    target = 1;
                end

                --print('current probe:' .. currentSequenceName .. ', gallery' .. galleryIdentities[galleryIdIndex] .. ', target: ' .. target .. ', ID index: ' .. identityIndices[galleryIdentities[galleryIdIndex]] .. '\n');
                --io.read()

                -- error calculation
                totalError = totalError + (criterion:forward(pred, target))
                                
                confusion:add(pred, target)
                --print('\n\npredicted :'); print(pred)
                --io.read()
                predCopy = torch.Tensor(3)
                predCopy[{{1,2}}]:copy(pred);
                predCopy[3] = identityIndices[galleryIdentities[galleryIdIndex]];

                scores[{{actualProbeImagesCount}, {actualGalleryIndex}, {}}] = predCopy;
                
                --only add the gallery index once (do not access the map unnecessarily)
                if(probeIdIndex == 1) then
                    galleryImageDetails[actualGalleryIndex] = galleryImagePath;
                end
                
                actualGalleryIndex = actualGalleryIndex + 1;

                -- lastIndex is counting the number of total images for helping with the index
                correctMatches[actualProbeImagesCount] = identityIndices[testdatanames[probeIdIndex]];
              end -- end for localGalleryImageIndex = 1
            end -- end for galleryIdIndex = 1

           -- increment the image index (lastIndex)
           probeImageDetails[actualProbeImagesCount] = allImgNames[probeImgIndex];
           actualProbeImagesCount = actualProbeImagesCount + 1;
        end

    end

    -- timing
    time = sys.clock() - time
    time = time / testDataCopied:size()
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
    print("\n==> actual probe images count = " .. (actualProbeImagesCount-1) .. ', calculated probe images count = ' .. totalProbeImages .. '\n')

    -- print confusion matrix
    print(confusion)

    -- update log/plot
    --testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    --if opt.plot then
    --    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    --    testLogger:plot()
    --end
    
    --save scores and correct matches
    matio.save(paths.concat(opt.save, 'scores.mat'), scores)
    matio.save(paths.concat(opt.save, 'correctMatches.mat'), correctMatches)
    
    --save gallery and probe details
    torch.save(paths.concat(opt.save, 'galleryImageDetails.t7'), galleryImageDetails)
    torch.save(paths.concat(opt.save, 'probeImageDetails.t7'), probeImageDetails)
    logger.info("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
    
    -- do the ranking for CMC (cumulative matching characteristics)
    cmc, topIndices = rankScores();
    
    -- print confusion matrix
    logger.info(confusion)
    
    -- next iteration:
    confusion:zero()
    
    return probeImageDetails, galleryImageDetails, topIndices, cmc, totalError;
end
