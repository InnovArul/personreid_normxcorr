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
package.path = "./modules/?.lua;" .. package.path
package.cpath = "./modules/?.so;" .. package.cpath
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
require 'NormCrossMapCorrelation'

-- classes
classes = {'same', 'different'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

opt = {}
torch.setdefaulttensortype('torch.FloatTensor') 

opt.useCuda = true; --true
opt.dataset = 'cuhk03'; -- cuhk03  |  others
opt.datasetname = 'cuhk03'  -- cuhk03 | cuhk01_test100  | cuhk01_test486   | qmulgrid
opt.datapath = '../datasets/' .. opt.datasetname .. '/' 
opt.dataType = 'detected' 
opt.testmode = 'test' -- test
opt.modelType = 'normxcorr'-- normxcorr | cin+normxcorr
opt.scale = {1}

MODEL_PATH = '../scratch/cuhk03/27-Nov-2016-17:53:15-personreiddummy_normxcorr_cuhk03_detected/normxcorr_cuhk03_detected#3.net'
--------------------------------------------------------------------------------------------------

opt.save = paths.dirname(MODEL_PATH)
MODEL_NAME = paths.basename(MODEL_PATH, paths.extname(MODEL_PATH))
opt.logFile = paths.concat(opt.save, MODEL_NAME .. '_forCMC.log')
opt.testErrorFile = paths.concat(opt.save, MODEL_NAME .. '_forCMC.eps')
logger.outfile = opt.logFile;
logger.trace(opt.save)
logger.trace(opt.logFile)
io.read()

----------------------------criterion definition---------------------------
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
-----------------------------------------------------------------------------

--load the appropriate data files
N = 10 -- number of tests
dofile 'dataForTests.lua';

dofile 'test.lua';
local errorHistory = nil
local epochHistory = nil

logger.trace('loading model : ' .. MODEL_PATH);
model = torch.load(MODEL_PATH)
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

