--[[
   doallTest.lua
   
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
require 'torch'
require 'optim'
require 'io'
require 'cutorch'
require 'nngraph'
require 'cunn'
require 'lfs'
require 'gnuplot'
logger = require 'log'
dofile 'utilities.lua'

-- classes
classes = {'same', 'different'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

opt = {}
torch.setdefaulttensortype('torch.FloatTensor') 
--cutorch.setDevice(2)

opt.useCuda = true; --true
opt.dataset = 'cuhk03'; -- ethz | cuhk03  |  others
opt.datasetname = 'cuhk03'
opt.datapath = '../personreid/datasets/dummy/' .. opt.datasetname .. '/'  -- datasets/ethz/ | datasets/cuhk03/  | datasets/viper/ | datasets/cuhk01_test100/ | datasets/cuhk01_test486/  | datasets/qmulgrid/
opt.dataType = 'detected' 
opt.testmode = 'test' -- validation | test
opt.modelType = 'normxcorr'-- 'cin+widersearch' -- normxcorr_smallersearch'; -- cin+xcorr | xcorr | multisiam | normxcorr | cin+xcorr_96maps | cin+normxcorr
opt.scale = {1}

rootLogFolder = paths.concat(lfs.currentdir(), 'CUHK03Training') --cuhk01_test486OnCUHK03
opt.save = paths.concat(rootLogFolder, '24Oct2016_personreiddummy_' .. opt.modelType .. '_' .. opt.datasetname .. '_' .. opt.dataType .. '_scale' .. table.tostring(opt.scale)) -- .. '_lr0.01batch160');

--model loop 
local modelIndexStart = 9
local modelIndexEnd= 9

opt.logFile = paths.concat(opt.save, opt.modelType.. '_' .. opt.datasetname .. '_' .. opt.dataType .. '_' .. opt.testmode .. '_start'.. modelIndexStart .. '_end' .. modelIndexEnd .. '_forCMC.log')
opt.testErrorFile = paths.concat(opt.save, opt.modelType.. '_' .. opt.datasetname .. '_' .. opt.dataType .. '_' .. opt.testmode .. '_start'.. modelIndexStart .. '_end' .. modelIndexEnd .. '_forCMC.eps')
logger.outfile = opt.logFile;
logger.trace(opt.save)
logger.trace(opt.logFile)
io.read()

----------------------------criterion definition---------------------------
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
-----------------------------------------------------------------------------

--load the appropriate data files
if(opt.testmode == 'validation') then
	N = 1  -- number of tests
	dofile 'data.lua';
else
	N = 10 -- number of tests
	dofile 'dataForTests.lua';
end

dofile 'test.lua';
local errorHistory = nil
local epochHistory = nil

for index = modelIndexStart, modelIndexEnd do

    --MODEL_NAME = paths.concat(opt.save, 'cin+normxcorr_CUHK03_labeled#12.net')   opt.dataset
    MODEL_NAME = paths.concat(opt.save, opt.modelType.. '_' .. opt.datasetname .. '_' .. opt.dataType .. '#' .. index .. '.net')

    logger.trace('loading model : ' .. MODEL_NAME);
    model = torch.load(MODEL_NAME)
    model:cuda()
    parameters,gradParameters = model:getParameters()   

    avgCMC = nil
    local avgError = 0

    for index = 1, N do
        -- start testing
        probeImageDetails, galleryImageDetails, topIndices, cmc, totalError = test();
        
        --allocate space for average rank holding buffer
        if(avgCMC == nil) then
          avgCMC = torch.Tensor(table.getn(galleryImageDetails)):fill(0)
        end
        
        avgCMC:add(cmc)
        avgError = avgError + totalError
    end

    logger.trace('average CMC:')
    avgCMC:div(N)
    avgError = avgError / N

    for index = 1, avgCMC:size(1) do
      logger.trace(index .. ' - ', avgCMC[index])
    end
    
    --append avgError and plot the error curve
    gnuplot.epsfigure(opt.testErrorFile)
    if(errorHistory == nil) then
        errorHistory = torch.Tensor({avgError})
        epochHistory = torch.Tensor({index})
    else
        errorHistory = torch.cat(errorHistory, torch.Tensor({avgError}), 1)
        epochHistory = torch.cat(epochHistory, torch.Tensor({index}), 1)
    end
    gnuplot.plot({'test-error', epochHistory, errorHistory})    
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('error')
    gnuplot.plotflush()    

end

