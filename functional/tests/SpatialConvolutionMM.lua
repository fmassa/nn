dofile 'init.lua'
require 'nn'

-- add it to a proper tester

--torch.setdefaulttensortype('torch.FloatTensor')

local precision = 1e-12

local input = torch.rand(1,1,50,50)
local gradOutput = torch.rand(1,1,50,50)

local module = nn.SpatialConvolutionMM(1,1,3,3,1,1,1,1)
SpatialConvolution_updateOutput(input,module.weight,module.bias,module.finput,module.output,module.kW,module.kH,module.dW,module.dH,module.padW,module.padH, module.nInputPlane, module.nOutputPlane)
local pred = module.output:clone()

module:forward(input)
assert((pred-module.output):abs():max() < precision, "Ops")

SpatialConvolution_updateGradInput(input,module.weight,module.bias,gradOutput,module.finput,module.fgradInput,module.gradInput,module.kW,module.kH,module.dW,module.dH,module.padW,module.padH)
local pred = module.gradInput:clone()
module:updateGradInput(input,gradOutput)

assert((pred-module.gradInput):abs():max() < precision, "Ops")

module:zeroGradParameters()
SpatialConvolution_accGradParameters(input,module.gradWeight,module.gradBias,gradOutput,module.finput,1)
local pred_weight = module.gradWeight:clone()
local pred_bias = module.gradBias:clone()
module:zeroGradParameters()
module:accGradParameters(input,gradOutput)

assert((pred_weight-module.gradWeight):abs():max() < precision, "Ops")
assert((pred_bias-module.gradBias):abs():max() < precision, "Ops")

print('All good !')
