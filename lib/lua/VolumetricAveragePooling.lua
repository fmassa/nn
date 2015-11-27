local VolumetricAveragePooling, parent = torch.class('THNN.VolumetricAveragePooling', 'nn.Module', THNN)

function VolumetricAveragePooling:__init(kT, kW, kH, dT, dW, dH)
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
end

function VolumetricAveragePooling:updateOutput(input)
  THNN.errcheck(
      'THNN_RealVolumetricAveragePooling_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.output:cdata(),
      self.kT, self.kW, self.kH,
      self.dT, self.dW, self.dH
   )
   return self.output
end

function VolumetricAveragePooling:updateGradInput(input, gradOutput)
   THNN.errcheck(
      'THNN_RealVolumetricAveragePooling_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.kT, self.kW, self.kH,
      self.dT, self.dW, self.dH
   )
   return self.gradInput
end

function VolumetricAveragePooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
end
