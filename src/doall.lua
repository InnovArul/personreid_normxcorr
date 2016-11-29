--[[
   doall.lua
   
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
require 'cutorch'
require 'lfs'
logger = require 'log'
dofile 'opts.lua';
dofile 'loss.lua';
dofile 'test.lua';
dofile 'utilities.lua'
require 'lfs'

epoch = 1;
    
logger.trace('PATH DETAILS:');
logger.trace(opt.save)
logger.trace(opt.logFile)
logger.trace(LOAD_MODEL_NAME)
logger.trace(SAVE_MODEL_NAME)
logger.trace('OPTIONS:');
for k,v in pairs(opt) do logger.trace(k,v) end

print("press <ENTER>, if the details are correct!");
io.read()

dofile 'data.lua';

--io.read()
if(opt.modelType == 'normxcorr' or opt.modelType == 'cin+normxcorr') then
    dofile ('model_' .. opt.modelType .. '.lua');
    dofile 'trainMultiGPU.lua';
else    
    print('unknown model type ' .. opt.modelType .. '\n\n');
    do return end;
end

--create or load the model
create_model();
--do return end

-- start training and testing
while (epoch <= 20) do
    train();
    
    --do return end
    test();
end

