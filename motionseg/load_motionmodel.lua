require 'nn';
require 'cunn';
require 'cudnn';

paths.dofile('DeepMaskAlexNet.lua');
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Helper script for loading model')
cmd:text()
cmd:option('-input', '', 'Path to input Torch model to be converted')
local config = cmd:parse(arg)

local model = torch.load(config.input);
print(model)
model = model:float()
model:evaluate()
