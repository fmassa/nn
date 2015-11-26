local Square, parent = torch.class('THNN.Square','nn.Module',THNN)

function Square:updateOutput(input)
   THNN.errcheck(
      'THNN_RealSquare_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.output:cdata()
   )
   return self.output
end

function Square:updateGradInput(input, gradOutput)
   THNN.errcheck(
      'THNN_RealSquare_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata()
   )
   return self.gradInput
end
