local Sigmoid = torch.class('THNN.Sigmoid', 'nn.Module', THNN)

function Sigmoid:updateOutput(input)
   THNN.errcheck(
     'THNN_RealSigmoid_updateOutput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     self.output:cdata()
   )
   return self.output
end

function Sigmoid:updateGradInput(input, gradOutput)
   THNN.errcheck(
     'THNN_RealSigmoid_updateGradInput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     gradOutput:cdata(),
     self.gradInput:cdata(),
     self.output:cdata()
   )
   return self.gradInput
end
