--[[
   train.lua
   
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

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'io'

require 'utilities'   -- user defined helper methods
logger = require 'log'

-------------------------------------------------------------------------
logger.trace '==> defining some tools'

local SAME = 1
local DIFFERENT = 2

-- classes
classes = {'same', 'different'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
trainLogger.showPlot = false; trainLogger.epsfile = paths.concat(opt.save, 'train.eps')

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger.showPlot = false; testLogger.epsfile = paths.concat(opt.save, 'train.eps')

----------------------------------------------------------------------
logger.trace '==> configuring optimizer'

if opt.optimization == 'CG' then
    optimState = {
        maxIter = opt.maxIter
    }
    optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
    optimState = {
        learningRate = opt.learningRate,
        maxIter = opt.maxIter,
        nCorrection = 10
    }
    optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
    optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = opt.learningRateDecay
    }
    optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
    optimState = {
        eta0 = opt.learningRate,
        t0 = trsize * opt.t0
    }
    optimMethod = optim.asgd

else
    error('unknown optimization method')
end

----------------------------------------------------------------------

logger.trace '==> defining training procedure'

function train()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    --make DataParallelTable 
    model = makeDataParallel(model, opt.nGPUs) 
    
    --do return end
    
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- get the handles for parameters and gradient parameters
    parameters,gradParameters = model:getParameters()
    
    datanames = table.getAllKeys(trainData['data']);
    allTrainData = trainData['data'];

    -- do one epoch
    logger.trace('==> doing epoch on training data:')
    logger.trace("==> online epoch # " .. epoch)

    totalTrainPersons = #datanames;

    if(opt.trainDataCount) then
        totalTrainPersons = opt.trainDataCount
    end
    
    -- create input pair combinations
    local inputs = {}
    local targets = {}
    numPositives = 0;
    numNegatives = 0;  
    totalFiles = 0;
          
    -- for all the training identites, create positive and twice the negative examples
    for t = 1,totalTrainPersons do

        --get current train sequence & collect determine all positive and negative sequences
        currentSequenceName = datanames[t];

        --insert all positive samples in inputs, targets
        allImgNames = table.getAllKeys(allTrainData[currentSequenceName])

        --TODO: check input pair generation
        for i = 1, #allImgNames do
            -- load new sample
            firstImage = allTrainData[currentSequenceName][allImgNames[i]];
            totalFiles = totalFiles + 1;
            
            for j = i+1, #allImgNames do
                secondImage = allTrainData[currentSequenceName][allImgNames[j]];
                table.insert(inputs, {firstImage, secondImage})
                table.insert(targets, SAME) --targets the 'same' class
                numPositives = numPositives + 1;
                
                -- insert 2 negative pairs
                for negIndex = 1, 2 do
                    negativeSampleSequence = allTrainData[datanames[getRandomNumber(1, #datanames, t)]];
                    allNegImgNames = table.getAllKeys(negativeSampleSequence)
                    negative = negativeSampleSequence[allNegImgNames[getRandomNumber(1, #allNegImgNames)]];
                    table.insert(inputs, {firstImage, negative})
                    table.insert(targets, DIFFERENT) --targets the 'different' class 
                    numNegatives = numNegatives + 1; 
                end                             
            end
        end
    end

     ---- after generating all positive, negative pairs, split them into batch size of opt.batchSize
    -- then for each batch size of opt.batchSize, do the stochastic gradient descent
    totalSamples = table.map_length(targets)
    totalBatches = totalSamples / opt.batchSize;

    -- if the totalSamples count is not divisble by opt.batchSize, then add +1
    if(totalSamples % opt.batchSize ~= 0) then
        totalBatches = math.floor(totalBatches + 1)
    end
        
    logger.debug('total pairs of training samples : ' .. totalSamples .. ' (total : ' .. totalFiles .. ' / positives: ' .. numPositives .. ' / negatives: ' .. numNegatives .. '), total batches: ' .. totalBatches)
    --io.read()

    -- randomize the generated inputs and outputs
    randomOrder = torch.randperm(totalSamples)

    --restrict batch count if requested
    if(opt.restrictedBatchCount and totalBatches > opt.restrictedBatchCount) then
        logger.trace ('\n\nrestricting batch count to ' .. opt.restrictedBatchCount .. '\n')
        totalBatches = opt.restrictedBatchCount;
    end

     ---- for each batch, do the SGD
    
    for batchIndex = 0, totalBatches - 1 do
        -- disp progress
        xlua.progress(batchIndex + 1, totalBatches)
        
        -- find the batchsamples start index and end index
        time = sys.clock()
        local batchStart = (batchIndex  * opt.batchSize) + 1
        local batchEnd = ((batchIndex + 1)  * opt.batchSize);

        -- make sure that index do not exceed the end index of the totalSamples
        if(batchEnd > totalSamples) then
            batchEnd = totalSamples
        end

        local currentBatchSize = batchEnd - batchStart + 1;

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            model:zeroGradParameters()

            -- f is the average of all criterions
            local f = 0
            currentTotalImagesTrained = 0;
            arrangedInputs = getArrangedInputsForNGPUs(inputs, targets, opt.nGPUs, batchStart, batchEnd, randomOrder)

            -- evaluate function for complete mini batch
            for i, trainset in ipairs(arrangedInputs) do

                input = trainset.input
                target = trainset.labels
                currentNumOfImages = target:size(1)
                if(opt.nGPUs == 1) then input = input[1]; target = target[1] end
                
                -- estimate f
                local output = localizeMemory(model:forward(input))
                local err = criterion:forward(output, target)
                f = f + err
                -- estimate df/dW
                local df_do = localizeMemory(criterion:backward(output, target))
                model:backward(input, df_do)
                
                currentTotalImagesTrained = currentTotalImagesTrained + currentNumOfImages

                -- update confusion
                if(opt.nGPUs > 1) then
                    confusion:batchAdd(output, target)
                else
                    confusion:add(output, target)
                end

             end --for i = batchStart, batchEnd do

            -- normalize gradients and f(X)
            --print('total images : ' .. currentTotalImagesTrained)
            gradParameters:div(currentTotalImagesTrained)
            f = f/currentTotalImagesTrained
           
            -- return f and df/dX
            return f,gradParameters
        end
        
        -- optimize on current mini-batch
        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
            optimMethod(feval, parameters, optimState)
        end
        
        -- DataParallelTable's syncParameters
       if model.needsSync then
          model:syncParameters()
       end        
       
        if(batchIndex % 500 == 0) then
            --print confusion matrix
            logger.trace(confusion)   
        end
    end -- for batchIndex = 0, totalBatches - 1 do


    -- time taken
    time = sys.clock() - time
    time = time / totalBatches
    logger.trace("\n\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- update logger/plot
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    if opt.plot then
        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        trainLogger:plot()
    end             

    -- logger.trace confusion matrix
    logger.trace(confusion)

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
      
    ----------------------------------------------------------------------
        -- save/log current net
    local filename = SAVE_MODEL_NAME .. '#' .. (epoch - 1) .. '.net';
    os.execute('mkdir -p ' .. sys.dirname(filename))
    logger.trace('==> saving model to '..filename)
    
    model = getInternalModel(model)    
    torch.save(filename, model)
end

--[[
-- logger.trace the size of the Threshold outputs
conv_nodes = model:findModules('nn.SpatialConvolutionMM')
for i = 1, #conv_nodes do
  --logger.trace(conv_nodes[i].output:size())
  --image.display(conv_nodes[i].output)
end
--]]
