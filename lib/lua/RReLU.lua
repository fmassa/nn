local ffi = require 'ffi'
local RReLU, parent = torch.class('THNN.RReLU', 'nn.Module', THNN)

function RReLU:__init(l, u, ip)
   parent.__init(self)
   self.lower = l or 1/8
   self.upper = u or 1/3
   assert(self.lower <= self.upper and self.lower >= 0 and self.upper >= 0)
   self.noise = torch.Tensor()
   self.train = true
   self.inplace = ip or false
end

function RReLU:updateOutput(input)
  local gen = ffi.typeof('THGenerator**')(torch._gen)[0]
  THNN.errcheck(
     'THNN_RealRReLU_updateOutput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     self.output:cdata(),
     self.noise:cdata(),
     self.lower,
     self.upper,
     self.train,
     self.inplace,
     gen
   )
   return self.output
end

function RReLU:updateGradInput(input, gradOutput)
   THNN.errcheck(
     'THNN_RealRReLU_updateGradInput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     gradOutput:cdata(),
     self.gradInput:cdata(),
     self.noise:cdata(),
     self.lower,
     self.upper,
     self.train,
     self.inplace
   )
   return self.gradInput
end

function RReLU:__tostring__()
  return string.format('%s (l:%f, u:%f)', torch.type(self), self.lower, self.upper)
end
