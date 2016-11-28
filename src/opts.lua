
package.path = "./modules/?.lua;" .. package.path
package.cpath = "./modules/?.so;" .. package.cpath
logger = require 'log'

opt = {}
--torch.setdefaulttensortype('torch.CudaTensor') 
cutorch.setDevice(1)

opt.useCuda = true; --true / false
opt.optimization = 'SGD'   -- CG  | LBFGS  |  SGD   | ASGD
opt.dataset = 'cuhk03'; -- cuhk03  |  others
opt.datasetname = 'cuhk03'
opt.datapath = '../datasets/' .. opt.datasetname .. '/'
opt.dataType = 'detected' -- labeled | detected
opt.testmode = 'validation' -- validation | test
opt.learningRate = 0.05
opt.weightDecay = 5e-4
opt.momentum = 0.9
opt.learningRateDecay = 1e-4
opt.batchSize = 128
opt.forceNewModel = true
opt.modelType = 'normxcorr' -- normxcorr | cin+normxcorr
opt.xnormcorrEps = 0.01
opt.traintype = ''
opt.plot = true
opt.GPU = 1
opt.nGPUs = 3

rootLogFolder = paths.concat(lfs.currentdir() .. '/../', 'scratch', opt.dataset) 
opt.save = paths.concat(rootLogFolder, os.date("%d-%b-%Y-%X-") .. 'personreid_' .. opt.modelType .. '_' .. opt.datasetname .. '_' .. opt.dataType);

print("options read from opts.lua")
dofile 'utilities.lua'

--if the save folder doesnot exist, create one
if(not isFolderExists(opt.save)) then
    paths.mkdir(opt.save)
end

opt.logFile = paths.concat(opt.save, opt.modelType.. '_' .. opt.datasetname .. '_' .. opt.dataType .. '.log')
logger.outfile = opt.logFile;

LOAD_MODEL_NAME = paths.concat(opt.save, opt.modelType.. '_' .. opt.datasetname .. '_' .. opt.dataType) 
SAVE_MODEL_NAME = paths.concat(opt.save, opt.modelType.. '_' .. opt.datasetname .. '_' .. opt.dataType)

