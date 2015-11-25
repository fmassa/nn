require 'torch'
THNN = {} -- improve it later
require 'nn'
local ffi = require 'ffi'

include 'ffi.lua'

-- we should replace all instances of THNN.NULL in the
-- function calls by THNN.getState()
-- still need to figure out if need to pass the tensor
-- type
THNN.NULL = ffi.NULL or nil
function THNN.getState()
  -- change it 
  return ffi.NULL or nil
end

local function errcheck(f, type, ...)
  -- handle different data types here
  local fname
  if type == 'torch.FloatTensor' then
    fname = string.gsub(f,'Real','Float')
  elseif type == 'torch.DoubleTensor' then
    fname = string.gsub(f,'Real','Double')
  else
    error('Type not supported: '..f)
  end
  local status = THNN.C[fname](...)
end
THNN.errcheck = errcheck

local function optionalTensor(t)
 if t then
    return t:cdata()
  else
    return THNN.NULL
  end
end
THNN.optionalTensor = optionalTensor

include 'Abs.lua'
include 'AbsCriterion.lua'
include 'ClassNLLCriterion.lua'
include 'DistKLDivCriterion.lua'
include 'HardShrink.lua'
include 'SpatialConvolutionMM.lua'
include 'SpatialMaxPooling.lua'

return THNN
