local nn = require 'nn'
local ffi = require 'ffi'
local THNN = require 'nn.lib.lua.env'

local AbsCriterion, parent = torch.class('THNN.AbsCriterion', 'nn.Criterion', THNN)

function AbsCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function AbsCriterion:updateOutput(input, target)
   local output = ffi.new 'double[1]'
   THNN.errcheck(
      'THNN_RealAbsCriterion_updateOutput', 
      input:type(), 
      THNN.NULL, 
      input:cdata(), 
      target:cdata(),
      self.sizeAverage,
      output
   )
   self.output = output[0]
   return self.output
end

function AbsCriterion:updateGradInput(input, target)
   THNN.errcheck(
      'THNN_RealAbsCriterion_updateOutput', 
      input:type(), 
      THNN.NULL, 
      input:cdata(), 
      target:cdata(),
      self.sizeAverage,
      self.gradInput:cdata()
   )
   return self.gradInput
end
