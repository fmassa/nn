local LogSoftMax = torch.class('THNN.LogSoftMax', 'nn.Module', THNN)

function LogSoftMax:updateOutput(input)
   THNN.errcheck(
     'THNN_RealLogSoftMax_updateOutput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     self.output:cdata()
   )
   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   THNN.errcheck(
     'THNN_RealLogSoftMax_updateGradInput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     gradOutput:cdata(),
     self.gradInput:cdata(),
     self.output:cdata()
   )
   return self.gradInput
end
