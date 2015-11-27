local SoftMax, _ = torch.class('THNN.SoftMax', 'nn.Module',THNN)

function SoftMax:updateOutput(input)
   THNN.errcheck(
      'THNN_RealSoftMax_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.output:cdata()
   )
   return self.output
end

function SoftMax:updateGradInput(input, gradOutput)
   THNN.errcheck(
      'THNN_RealSoftMax_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   return self.gradInput
end
