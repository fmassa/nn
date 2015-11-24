require 'torch'
local THNN = require 'nn.lib.lua.env'
local ffi = require 'ffi'

include 'ffi.lua'

THNN.NULL = ffi.NULL or nil

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
include 'SpatialConvolutionMM.lua'

return THNN
