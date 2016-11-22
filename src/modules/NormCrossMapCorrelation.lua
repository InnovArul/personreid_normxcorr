require 'nn'
local NormCrossMapCorrelation, parent = torch.class('nn.NormCrossMapCorrelation', 'nn.Module');

require 'io'
require 'cutorch'
require 'torch'
ffi = require("ffi")
ffi.cdef[[
    void updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, int patchwidth, int verticalWidth, 
                      THCudaTensor *meanMaps, THCudaTensor *stdMaps);
    void updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *gradOutput, 
                        THCudaTensor *gradInput, int patchwidth, int verticalWidth, THCudaTensor *meanMaps, THCudaTensor *stdMaps);
]]

--constructor for NormCrossMapCorrelation
function NormCrossMapCorrelation:__init(patchwidth, verticalWidth)
    parent.__init(self)
    
    self.patchwidth = patchwidth
    self.verticalWidth = verticalWidth
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.gradOutput = torch.Tensor()
    self.meanMaps = torch.Tensor()
    self.stdMaps = torch.Tensor()
end

--[[
   
   name: updateOutput
   @param input - 50 layers of 37 x 12 patches
   @return - output of tensor of 1500 layers of 37x12
   
]]--
-- override the predefined methods
function NormCrossMapCorrelation:updateOutput(input)
  local cutorchState = cutorch.getState()
  cbind = ffi.load(paths.dirname(paths.thisfile()) .. "/libNormCrossMapCorrelation.so");
  cbind.updateOutput(cutorchState,
                    input:cdata(), 
                    self.output:cdata(),
                    self.patchwidth,
                    self.verticalWidth,
                    self.meanMaps:cdata(),
                    self.stdMaps:cdata());
   
  cbind = ffi.NULL;

  return self.output;
end

-- API to determine the gradient of output w.r.t., input
function NormCrossMapCorrelation:updateGradInput(input, gradOutput)
  self.gradOutput:resizeAs(gradOutput):copy(gradOutput);
  local cutorchState = cutorch.getState()
  cbind = ffi.load(paths.dirname(paths.thisfile()) .. "/libNormCrossMapCorrelation.so");
  
  cbind.updateGradInput(cutorchState,
                      input:cdata(),
                      self.output:cdata(),
                      self.gradOutput:cdata(),
                      self.gradInput:cdata(), 
                      self.patchwidth, 
                      self.verticalWidth,
                      self.meanMaps:cdata(),
                      self.stdMaps:cdata());
                      
  cbind = ffi.NULL;

  return self.gradInput;
end

--[[]
function NormCrossMapCorrelation:write(f)
   local var = {}
   for k,v in pairs(self) do
     if(k ~= "cbind") then var[k] = v end
   end
   f:writeObject(var)
end

--[[
function NormCrossMapCorrelation:clone()
   clone = parent.clone(self)
   clone.cbind = ffi.load(paths.dirname(paths.thisfile()) .. "/libNormCrossMapCorrelation.so");
   return clone
end 
--]]
