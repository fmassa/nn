local VolumetricConvolution, parent = torch.class('THNN.VolumetricConvolution', 'nn.Module', THNN)

function VolumetricConvolution:__init(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
   parent.__init(self)

   dT = dT or 1
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kT = kT
   self.kW = kW
   self.kH = kH
   self.dT = dT
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   -- temporary buffers for unfolding (CUDA)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()   
   self:reset()
end

function VolumetricConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kT*self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function VolumetricConvolution:updateOutput(input)
   THNN.errcheck(
      'THNN_RealVolumetricConvolution_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.dT, self.dW, self.dH
   )
   return self.output
end

function VolumetricConvolution:updateGradInput(input, gradOutput)
   THNN.errcheck(
      'THNN_RealVolumetricConvolution_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.dT, self.dW, self.dH,
      self.nOutputPlane
   )
   return self.gradInput
end

function VolumetricConvolution:accGradParameters(input, gradOutput, scale)
   THNN.errcheck(
      'THNN_RealVolumetricConvolution_accGradParameters',
      input:type(),
      THNN.NULL,
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self.gradBias:cdata(),
      self.dT, self.dW, self.dH,
      self.nOutputPlane,
      scale or 1
   )
end
