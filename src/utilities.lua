--[[
   utilities.lua
   
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

require 'lfs'
require 'image'
require 'torch'
require 'cutorch'
--require 'cunn'
local ffi=require 'ffi'
require 'xlua'
require 'DataParallelTableForSiamese'

STRIDE = 5

-- for cin+xcorr
PATCHSIZE = 5
VERTICALWIDTH = 5

-- for cin+xcorr5+xcorr3
PATCHSIZE1 = 5
VERTICALWIDTH1 = 5
PATCHSIZE2 = 3
VERTICALWIDTH2 = 3

LAYERS = 50

-- set the random number seed
math.randomseed( os.time() )

local Threads = require 'threads'
local t = Threads(10,
                  function()
                    require 'image'
                    require 'io'
                    opt = opt
                  end
                ) -- create a pool of threads

--[[
   
   name: makeDataParallel
   @param
   @return creates Parallel models incase if we are using multiple GPUs
   
]]--

model_single = {}
function makeDataParallel(model, nGPU)
    -- if the number of GOUs used is more than 1,
    -- create DataParallelTableForSiamese
    if nGPU > 1 then
      print('converting module to nn.DataParallelTableForSiamese')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      model_single = model
      model = nn.DataParallelTableForSiamese(1) --, true, true
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
      
      -- incase, if we are using DataParallelTableForSiamese, create threads to avoid delay in kernel launching
      model:threads(function(idx)
          package.path = "./modules/?.lua;" .. package.path
          package.cpath = "./modules/?.so;" .. package.cpath
          require 'nn'
          require 'cunn'
          require 'nngraph'
          require 'cutorch'
          require 'NormCrossMapCorrelation'
          require 'CrossInputNeighborhood'
          
          cutorch.setDevice(idx)
      end);         
    end
    
    cutorch.setDevice(opt.GPU)

    return model
end


--[[
   
   name: sliceRange
   @param
   @return calculate the range of indices needed based on index and number of total splits
   
]]--
function sliceRange(nElem, idx, splits)
    -- calculate the count of common elements for all the GPUs
   local commonEltsPerMod = math.floor(nElem / splits)
   
    -- calculate the count of reamining elements for which the element-count shall be commonEltsPerMod + 1
   local remainingElts = nElem - (commonEltsPerMod * splits)
   
   -- according to current idx, how much "commonEltsPerMod + 1" elements are there?
   local commonPlusOneEltsCount = math.min(idx - 1, remainingElts)
   -- according to current idx, how much "commonEltsPerMod" elements are there?
   local commonEltsCount = (idx - 1) - commonPlusOneEltsCount 
   
   -- determine the start index
   local rangeStart = (commonPlusOneEltsCount * (commonEltsPerMod + 1)) + 
                        (commonEltsCount * commonEltsPerMod) + 1
                        
    -- determine the total elements for current index
   local currentElts = commonEltsPerMod
   if(idx <= remainingElts) then currentElts = commonEltsPerMod + 1 end

    -- return start index and elements count
   return rangeStart, currentElts
end


--[[
   UNUSED AS OF NOW
   name: cleanDPT
   @param
   @return clear the DataParallelTableForSiamese and return new DataParallelTableForSiamese
   
]]--
local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTableForSiamese(1)
   cutorch.setDevice(opt.GPU)
   newDPT:add(module:get(1), opt.GPU)
   return newDPT
end

--[[
   
   name: saveDataParallel
   @param
   @return save the model with given filename
   
]]--
function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTableForSiamese' then
      -- save the model in first GPU
      temp_model = model:get(1):clearState()
      torch.save(filename, temp_model)
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTableForSiamese' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else -- incase the given model is a plain model (due to nGPUs = 1)
      temp_model = model:clearState()
      torch.save(filename, temp_model)
   end
end

--[[
   UNUSED AS OF NOW
   name: loadDataParallel 
   @param
   @return load the saved DataParallelTableForSiamese model
   
]]--
function loadDataParallel(filename, nGPU)
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTableForSiamese' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTableForSiamese' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTableForSiamese module.')
   end
end


function cudaTheData(data)
    local clonedData = nil
    
    -- if the data is really a table, then recurse into table and cuda the data
    -- remember that, the criterions and models in torch are even table, 
    -- but torch.typename will give valid class labels for them.
    -- so, we can differentiate the real tables with torch.typename
    if('table' == type(data) and (nil == torch.typename(data))) then
        clonedData = {}
        for i, internalData in ipairs(data) do
            clonedData[i] = cudaTheData(internalData)
        end
    else
        clonedData = data:cuda()
    end
    return clonedData
end

--[[

   name: getArrangedInputsForNGPUs 
   @param
   @return split the data into batches based on number of GPUs used
]]--
function getArrangedInputsForNGPUs(data, target, nGPUs, beginIndex, endIndex, randomOrder)
    local batches = {}
    local totalCount = 0
    if beginIndex and endIndex then 
      totalCount = endIndex - beginIndex + 1
    else
      totalCount = table.getn(target)
      beginIndex = 1
      endIndex = totalCount
    end
    
    local totalBatches = math.floor(totalCount / nGPUs)
    if(totalCount % nGPUs ~= 0) then totalBatches = totalBatches + 1 end
    --logger.trace('total number of inputs: ' .. totalCount .. ', batches : ' .. totalBatches)
    
    for index = 1, totalBatches do
        local startIndex, count = sliceRange(totalCount, index, totalBatches)

        local currentData = {}
        
        --remember that targets are always in default device index
        local currentTargets = torch.CudaTensor(count)
        --xlua.progress(index, totalBatches)
        
        --get previous default dev index
        local prevDevId = cutorch.getDevice()
        for dataIndex = 1, count do
           currentDataIndex = randomOrder[(beginIndex - 1) + (startIndex- 1) + dataIndex]
           
           --copy to correct GPU
           local clonedData = {}
           cutorch.withDevice(dataIndex, function() 
                                            --for i, internalData in ipairs(data[currentDataIndex]) do
                                               --copy the data to particular GPU
                                            --   clonedData[i] = internalData:cuda()
                                               --print(clonedData[i]:getDevice())
                                            --end
                                            clonedData = cudaTheData(data[currentDataIndex])
                                        end)
      
           --maybe, involve multi-threading?
           
           currentData[dataIndex] = clonedData
           
           currentTargets[dataIndex] = target[currentDataIndex] 
        end
        --set the default device index
        cutorch.setDevice(prevDevId);

        batches[index] = {
            input=currentData,
            labels=currentTargets
        }
    end

    return batches
end

--[[

   name: getInternalModel 
   @param
   @return get the internal model from wrapped model, if DataParallelTableForSiamese is used
]]--
function getInternalModel(model)
    currentModel = model
    if torch.type(model) == 'nn.DataParallelTableForSiamese' then
        currentModel = model:get(1)
    end
    return currentModel
end

function isFolderExists(strFolderName)
	if lfs.attributes(strFolderName:gsub("\\$",""),"mode") == "directory" then
		return true
	else
		return false
	end
end

--[[
copy the table's first level values
--]]
function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

--[[
copy the table with all the internal key and value pairs 
--]]
function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

--[[
   
   name: loadImageWithScale
   @param
   @return
   
]]--
function loadImageWithScale(filepath, scale)
    if(scale == nil) then scale = opt.scale end
    local img = image.load(filepath)
    local imgOutput = img
    
    --incase if the scale is not nil or 1, do the gaussian pyramid process
    if(scale ~= nil) then
        -- if opt.scale is not a table, then make it as a table
        if(not(type(scale) == 'table')) then 
            scale = {scale} 
        end
        
        local totalScales = table.getn(scale)
        
        -- if the only scale needed is 1, then no need for gaussianpyramid
        if(not(totalScales == 1 and scale[1] == 1)) then
            imgOutput = image.gaussianpyramid(img, scale)
            
            --if there is only one scale needed (such as 0.5), then get the image from the output table from image.gaussianpyramid
            if(totalScales == 1) then imgOutput = imgOutput[1] end
        end
    end
    
    return imgOutput
end

--[[
   
   name: loadAllImagesFromFolders
   @param
   @return
   
]]--

function loadAllImagesFromFolders(fileNames, filePaths)
	loadedfiles = {};
--[[]    
    for i, foldername in ipairs(fileNames) do
        local hashKey = i .. '-' .. foldername;
        print(foldername .. ' --> ' .. filePaths[i])
        logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
        allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
        
        local imgTable = {};
        for j, imgname in ipairs(allImgPaths) do 
           local loadImageWithScale = loadImageWithScale
           local scale = opt.scale
           
           t:addjob(
                function() 
                    local loadedImg = (loadImageWithScale(imgname, scale))
                    return loadedImg
                end,
                
                function(img)
                    imgTable[imgname] = img; 
                end
           )
        end
        logger.trace('\tNumber of images : '.. #allImgPaths)
        loadedfiles[hashKey] = imgTable;
    end
    
    t:synchronize()
    return loadedfiles
    --]]

	loadedfiles = {};
    
    for i, foldername in ipairs(fileNames) do
        print(foldername .. ' --> ' .. filePaths[i])
        logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
        allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
        
        imgTable = {};
        for j, imgname in ipairs(allImgPaths) do 
           img = (image.load(imgname)) -- '1' --
           imgTable[imgname] = img; 
        end
        logger.trace('\tNumber of images : '.. #allImgPaths)
        loadedfiles[foldername] = imgTable;
    end
    
    return loadedfiles
    
end

--[[
   
   name: getAllFiles
   @param
   @return
   
]]--

function getAllFileNamesInDir(directory)

    index = 1;
    filesOrFolderNames = {};
    filesOrFolderPaths = {};
    
    if(lfs.attributes(directory) == nil) then
      logger.warn('given dorectory ' .. directory .. ' does not exist!');
      return {}, {};
    end
    
    for file in lfs.dir(directory) do
        --print (file )
        if(file:sub(1,1) ~= '.' and file ~= '..') then
            filesOrFolderNames[index] = file;
            filesOrFolderPaths[index] = directory .. '/' .. file;
            index = index + 1;
        end
    end

    return filesOrFolderNames, filesOrFolderPaths;
end

--[[
   
   name: table.val_to_str
   @param
   @return string formatted value
   
]]--
function table.val_to_str ( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table.tostring( v ) or
      tostring( v )
  end
end

--[[
   
   name: table.key_to_str
   @param
   @return string formatted key
   
]]--
function table.key_to_str ( k )
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. table.val_to_str( k ) .. "]"
  end
end

--[[
   
   name: table.tostring
   @param
   @return string formatted hash
   
]]--

function table.tostring( tbl)
  -- return tbl, if it is not type of table
  if('table' ~= type(tbl)) then
    return tbl
  end
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, table.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
    end
  end
  return "{" .. table.concat( result, "," ) .. "}"
end

--[[
   
   name: table.map_length
   @param
   @return the number of keys in the table (i.e., effectively the length of the table)
   
]]--

function table.map_length(t)
    local c = 0
    for k,v in pairs(t) do
         c = c+1
    end
    return c
end


--[[
   
   name: table.getAllKeys
   @param
   @return the number of keys in the table (i.e., effectively the length of the table)
   
]]--

function table.getAllKeys(tbl)
    local keyset={}
    local n=0

    for k,v in pairs(tbl) do
      n=n+1
      keyset[n]=k
    end
    
    table.sort(keyset)
    return keyset;
end


--[[
   
   name: table.getValues
   @param
   @return get all the values for given keys
   
]]--

function table.getValues(tbl, keys)
    local values={}
    local n=0

    for index = 1, keys:size(1) do
      values[index]=tbl[keys[index]]
    end
    
    table.sort(values)
    return values;
end


--[[
   
   name: getRandomNumber
   @param
   @return a random number between lowe and upper, but without the number in exclude
   
]]--

function getRandomNumber(lower, upper, exclude)
    randNumber = math.random(lower, upper);
    while(randNumber == exclude) do
        randNumber = math.random(lower, upper);
    end
    return randNumber;
end


--[[
   
   name: localizeMemory
   @param
   @return copies the given tensor to GPU, incase GPU usage is forced
   
]]--
function localizeMemory(tensor)
  if(opt.useCuda) then
     newTensor = cudaTheData(tensor)
  else
    newTensor = tensor;
  end
  
  return newTensor;
end

--[[

getTopNIndices: get the indices according to the sorted indices from rankScores.lua

]]--
function getTopNIndices(indices, topN, actualListOfIndices)
  actualTopIndices = torch.Tensor(topN);
  actualTopRawIndices = torch.Tensor(topN);
  
  for index = 1, topN do
    actualTopIndices[index] = actualListOfIndices[indices[1][index]][1];
    actualTopRawIndices[index] = indices[1][index];
  end
  
  return actualTopIndices, actualTopRawIndices
end

--[[
   
   name: localizeModel
   @param
   @return copies the given model to GPU, incase GPU usage is forced
   
]]--
function localizeModel(model)
  if(opt.useCuda) then
     model = model:cuda();
  end
  return model
 end


--[[
   
   name: allocateTensor
   @param
   @return allocates the tensor of the given size in GPU or CPU based on the option set
   
]]--
function allocateTensor(givenSize)
  if(opt.useCuda) then
     newTensor = torch.CudaTensor(unpack(givenSize));
  else
     newTensor = torch.Tensor(unpack(givenSize));
  end
  
  return newTensor;
end

local flowUnit = 10
local EMDSummary_Start = flowUnit + 1
local EMDSummary_End = EMDSummary_Start + LAYERS - 1

function copyGndDistancesFromFlowCalcToConv()
    gndDistance1 = model.modules[flowUnit].groundDistance1;
    gndDistance2 = model.modules[flowUnit].groundDistance2;

    for index = EMDSummary_Start, EMDSummary_End do
        currIndex = index - (EMDSummary_Start - 1);
        --print(currIndex)

        if(currIndex <= LAYERS/2) then 
            model.modules[index].weight[1]:copy(gndDistance1[currIndex]);
            model.modules[index].bias = localizeMemory(torch.Tensor({0}))
        else
            model.modules[index].weight[1]:copy(gndDistance2[currIndex - LAYERS/2]);
            model.modules[index].bias = localizeMemory(torch.Tensor({0}))
        end
    end
end

function copyGndDistancesFromConvToFlowCalc()

   -- collect all the convolution kernels (weights) from EMD summary units
    convolutionKernels = {}
    
    for index = EMDSummary_Start, EMDSummary_End do
        convolutionKernels[index - (EMDSummary_Start - 1)] = model.modules[index].weight[1];
    end
    
    groundDistanceCreator = nn.JoinTable(1)
    groundDistances = groundDistanceCreator:forward(convolutionKernels)

    model.modules[flowUnit].groundDistance1 = localizeMemory(groundDistances[{{1, LAYERS/2}, {}, {}}]);
    model.modules[flowUnit].groundDistance2 = localizeMemory(groundDistances[{{LAYERS/2, LAYERS}, {}, {}}]);
end

