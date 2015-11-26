local Sqrt, parent = torch.class('THNN.Sqrt','nn.Module',THNN)

function Sqrt:updateOutput(input)
   THNN.errcheck(
      'THNN_RealSqrt_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.output:cdata()
   )
   return self.output
end

function Sqrt:updateGradInput(input, gradOutput)
   THNN.errcheck(
      'THNN_RealSqrt_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   return self.gradInput
end
