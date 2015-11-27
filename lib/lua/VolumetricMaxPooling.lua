local VolumetricMaxPooling, parent = torch.class('THNN.VolumetricMaxPooling', 'nn.Module', THNN)

function VolumetricMaxPooling:__init(kT, kW, kH, dT, dW, dH)
   parent.__init(self)

   dT = dT or kT
   dW = dW or kW
   dH = dH or kH
   
   self.kT = kT
   self.kH = kH
   self.kW = kW
   self.dT = dT
   self.dW = dW
   self.dH = dH

   self.indices = torch.Tensor()
end

function VolumetricMaxPooling:updateOutput(input)
   THNN.errcheck(
      'THNN_RealVolumetricMaxPooling_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.output:cdata(),
      self.kT, self.kW, self.kH,
      self.dT, self.dW, self.dH,
      self.indices:cdata()
   )
   return self.output
end

function VolumetricMaxPooling:updateGradInput(input, gradOutput)
   THNN.errcheck(
      'THNN_RealVolumetricMaxPooling_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.dT, self.dW, self.dH,
      self.indices:cdata()
   )
   input.nn.VolumetricMaxPooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function VolumetricMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
