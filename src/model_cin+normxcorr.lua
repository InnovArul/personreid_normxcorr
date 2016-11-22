--[[
   model_cin+normxcorr.lua
   
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

require 'torch';
require 'nn';
require 'io';
require 'lfs'
require 'cunn'
require 'cutorch'
dofile 'utilities.lua'
require 'nngraph';
require 'NormCrossMapCorrelation'
require 'CrossInputNeighborhood'

--[[
opt = {}
opt.forceNewModel = true
opt.useCuda = true
opt.logFile = 'modelCreation.log'
]]--

logger = require 'log'
logger.outfile = opt.logFile

--cudnn.benchmark = true;
--cudnn.fastest = true;
--cudnn.verbose = true;

--define fillcolors for different layers
COLOR_CONV = 'cyan';
COLOR_MAXPOOL = 'grey';
COLOR_RELU = 'lightblue';
COLOR_SOFTMAX = 'green';
COLOR_FC = 'orange';
COLOR_CROSSMAP = 'yellow';
COLOR_AUGMENTS = 'brown';

TEXTCOLOR = 'black';
NODESTYLE = 'filled';

-- input dimensions:
local nfeats = 3
local width = 160
local height = 60

-- hidden units, filter sizes (for ConvNet only):
local nstates = {20,25,25,500}
local filtsize = {5,5,3,3}
local poolsize = 2
local padsize = 0

nnpackage = nn;

-- read the image
--img = image.load('ahmed_model/1.png');

function create_model()
    nngraph.setDebug(true);
    ---check if an already saved model present in the current directory
    --if so, load the model and continue to train and test

    if(opt.forceNewModel == nil or opt.forceNewModel ~= false ) then
      LOAD_MODEL_NAME = 'any'
    end
    
    if(not opt.forceNewModel or lfs.attributes(LOAD_MODEL_NAME) ~= nil) then
        logger.trace('\n loading the existing model : ' .. LOAD_MODEL_NAME .. '\n')
        model = torch.load(LOAD_MODEL_NAME)
        model = localizeMemory(model);
        io.read()
    else
        logger.trace('\n creating new model : ' .. LOAD_MODEL_NAME .. '\n')
        --------------------------------------------------------------------------------
        -- Network for the image-1  (called as subNetwork1)
        --------------------------------------------------------------------------------
        --subNetwork1 Tied convolution maxpooling-I
        img1_conv1 = nnpackage.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1])():annotate{
            name='Image[1] - Convolution unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        img1_relu1 = nnpackage.ReLU()(img1_conv1):annotate{
            name='Image[1] - ReLU unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --Maxpool
        img1_maxpool1 = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(img1_relu1):annotate{
            name='Image[1] - Maxpooling unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        --subNetwork1 Tied convolution maxpooling-II
        img1_conv2 =nnpackage.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize[2])(img1_maxpool1):annotate{
            name='Image[1] - Convolution unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        img1_relu2 = nnpackage.ReLU()(img1_conv2):annotate{
            name='Image[1] - ReLU unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --Maxpool
        img1_maxpool2 = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(img1_relu2):annotate{
            name='Image[1] - Maxpooling(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        -------------------------------------------------------------------
        -- Network for the image-2 (called as subNetwork2
        --------------------------------------------------------------------------------
        --subNetwork2 Tied convolution maxpooling-I
        img2_conv1 = nnpackage.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1])():annotate{
            name='Image[2] - Convolution unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --share the weights with image1 conv1 layer
        img2_conv1.data.module:share(img1_conv1.data.module, 'weight', 'bias', 'gradWeight', 'gradBias');

        --ReLU
        img2_relu1 = nnpackage.ReLU()(img2_conv1):annotate{
            name='Image[2] - ReLU unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --Maxpool
        img2_maxpool1 = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(img2_relu1):annotate{
            name='Image[2] - Maxpooling unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        --subNetwork2 Tied convolution maxpooling-II
        img2_conv2 = nnpackage.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize[2])(img2_maxpool1):annotate{
            name='Image[2] - Convolution unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        img2_relu2 = nnpackage.ReLU()(img2_conv2):annotate{
            name='Image[2] - ReLU unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --share the weights with image1 conv2 layer
        img1_conv2.data.module:share(img2_conv2.data.module, 'weight','bias','gradWeight','gradBias');

        --Maxpool
        img2_maxpool2 = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(img2_relu2):annotate{
            name='Image[2] - Maxpooling unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        -----------------------------------------------------------------------------------------------
        --Join the two parallel networks for cross input neighborhood differences layer processing
        -----------------------------------------------------------------------------------------------

        -- Joining layer to join the filtered features from two subNetworks
        m3 = nn.JoinTable(1)({img1_maxpool2, img2_maxpool2}):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };
        
        --cross CorrelationUnit unit
        normCrossCorrelationUnit = nn.NormCrossMapCorrelation(PATCHSIZE, VERTICALWIDTH)(m3):annotate{
            name='Normalized Cross Map correlation unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CROSSMAP}
        };
        
        CCU_ReLU = nn.ReLU()(normCrossCorrelationUnit):annotate{
            name='Normalized Cross Map correlation unit - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --nstates[3], nstates[3]
        spatialConfidence = nn.SpatialConvolution(12*5*25, 25, 1, 1)(CCU_ReLU):annotate{
            name='Normalized Cross Map correlation confidence summary unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        --ReLU
        CCU_summary_relu = nnpackage.ReLU()(spatialConfidence):annotate{
            name='Patch summary features[1] - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };        
                
        global_summary = nn.SpatialConvolution(nstates[3], nstates[3], 3, 3)(CCU_summary_relu):annotate{
            name='Normalized Cross Map correlation summary unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        global_summary_maxpool = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(global_summary):annotate{
            name='Normalized Cross Map correlation summary maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        --reshape
        reshape = nn.Reshape(nstates[3]*17*5)(global_summary_maxpool):annotate{
            name='Reshaping unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --FC500
        FC500_XCORR = nn.Linear(nstates[3]*17*5, nstates[4])(reshape):annotate{
            name='Fully connected layer - 500 nodes',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_FC}
        };

    -------------------------------------------------------------------------------------------------
    -- Ahmed paper branch
       -- Joining layer to join the filtered features from two subNetworks
        m3 = nn.JoinTable(1)({img1_maxpool2, img2_maxpool2}):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --Cross input neighborhood differences
        CIN = nn.CrossInputNeighborhood()(m3):annotate{
            name='Cross Input Neighborhood unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --join
        joining = nn.JoinTable(1)(CIN):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };
        
        --CrossInputNeighborhood ReLU
        CINrelu = nnpackage.ReLU()(joining):annotate{
            name='Cross Input Neighborhood unit - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        --Patch summary features
        patch_summary = nnpackage.SpatialConvolution(1250, 25, 1, 1)(CINrelu):annotate{
            name='Patch summary features[1] - Convolution unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        patch_summary_relu = nnpackage.ReLU()(patch_summary):annotate{
            name='Patch summary features[1] - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        --Across patch features
        across_patch = nnpackage.SpatialConvolution(25, 25, 3, 3)(patch_summary_relu):annotate{
            name='Across patch features[1] - Convolution unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --Maxpool
        across_maxpool = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(across_patch):annotate{
            name='Across patch features[1] - Maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        ----------------------------------------------------------------------
        -- join -> reshape -> FC500
        ----------------------------------------------------------------------------------------------------

        --reshape
        reshape = nn.Reshape(25*17*5)(across_maxpool):annotate{
            name='Reshaping unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --FC500
        FC500_CIN = nn.Linear(25*17*5, 500)(reshape):annotate{
            name='Fully connected layer - 500 nodes',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_FC}
        };

        FC_Join = nn.JoinTable(1)({FC500_XCORR, FC500_CIN});
        
        --ReLU
        FCReLU = nnpackage.ReLU()(FC_Join):annotate{
            name='Fully connected layer - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        --FC2
        FC2 = nn.Linear(1000, 2)(FCReLU):annotate{
            name='Fully connected layer - 2 nodes',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_FC}
        };

        --Softmax
        result = nnpackage.LogSoftMax()(FC2):annotate{
            name='Softmax unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_SOFTMAX}
        };

        print ('model created\n');
        
        -- packaging the network into gModule (graphical module) from nngraph
        --
        model = localizeMemory(nn.gModule({img1_conv1, img2_conv1}, {result}));
        model = localizeMemory(model);
        
        graph.dot(model.fg, 'model', 'personreid_cin+normxcorr')
        
        --do return end
        --
        --CHECKING THE WEIGHT SHARING BY PRINTING THE POINTERS!
        logger.trace('img1 conv1')
        logger.trace(model.modules[1].weight:data()) -- img1 conv1
        logger.trace('img1 conv2')
        logger.trace(model.modules[4].weight:data()) -- img1 conv2
        logger.trace('img2 conv1')
        logger.trace(model.modules[7].weight:data()) -- img2 conv1
        logger.trace('img2 conv2')
        logger.trace(model.modules[10].weight:data()) -- img2 conv2

        --img = localizeMemory(image.scale(img, 60, 160));
        
        -- forward for New model
        --pred = localizeMemory(model:forward({img, img})); 
 
     end -- if(path.exists(MODEL_NAME))
--]]

  -- Retrieve parameters and gradients:
  -- this extracts and flattens all the trainable parameters of the mode
  -- into a 1-dim vector
  -- for future use in train.lua

    parameters,gradParameters = model:getParameters()
    --graph.dot(model.fg, 'model', 'personreid')

end

--create_model()
