local LogSigmoid, parent = torch.class('THNN.LogSigmoid', 'nn.Module', THNN)

function LogSigmoid:__init()
   parent.__init(self)
   self.buffer = torch.Tensor()
end

function LogSigmoid:updateOutput(input)
   THNN.errcheck(
     'THNN_RealLogSigmoid_updateOutput',
     input:type(),
     THNN.NULL,
     input:cdata(),
     self.output:cdata(),
     self.buffer:cdata()
   )
   return self.output
end

function LogSigmoid:updateGradInput(input, gradOutput)
   THNN.errcheck(
     'THNN_RealLogSigmoid_updateGradInput',
     input:type(),
     THNN.NULL,
     gradOutput:cdata(),
     self.gradInput:cdata(),
     self.buffer:cdata()
   )
   return self.gradInput
end
