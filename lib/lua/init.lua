require 'torch'
THNN = {}--require 'THNN.env'
include 'ffi.lua'
local ffi = require 'ffi'

local errcheck = function(f, type, ...)
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

include 'SpatialConvolutionMM.lua'

return THNN
