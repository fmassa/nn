local PReLU, parent = torch.class('THNN.PReLU','nn.Module', THNN)

function PReLU:__init(nOutputPlane)
   parent.__init(self)
   -- if no argument provided, use shared model (weight is scalar)
   self.nOutputPlane = nOutputPlane or 0
   self.weight = torch.Tensor(nOutputPlane or 1):fill(0.25)
   self.gradWeight = torch.Tensor(nOutputPlane or 1)
   self.gradWeightBuf = torch.Tensor()
   self.gradWeightBuf2 = torch.Tensor()
end

function PReLU:updateOutput(input)
   THNN.errcheck(
     'THNN_RealPReLU_updateOutput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     self.output:cdata(),
     self.weight:cdata(),
     self.nOutputPlane
   )
   return self.output
end

function PReLU:updateGradInput(input, gradOutput)
   THNN.errcheck(
     'THNN_RealPReLU_updateGradInput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     gradOutput:cdata(),
     self.gradInput:cdata(),
     self.weight:cdata(),
     self.nOutputPlane
   )
   return self.gradInput
end

function PReLU:accGradParameters(input, gradOutput, scale)
   THNN.errcheck(
     'THNN_RealPReLU_accGradParameters',
     input:type(),
     THNN.NULL,
     input:cdata(),
     gradOutput:cdata(),
     self.gradInput:cdata(),
     self.weight:cdata(),
     self.gradWeight:cdata(),
     self.nOutputPlane,
     scale or 1
   )
   return self.gradWeight
end
