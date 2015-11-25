local HardShrink, parent = torch.class('THNN.HardShrink', 'nn.Module', THNN)

function HardShrink:__init(lam)
   parent.__init(self)
   self.lambda = lam or 0.5
end

function HardShrink:updateOutput(input)
   THNN.errcheck(
     'THNN_RealHardShrink_updateOutput', 
     input:type(), 
     THNN.NULL, 
     input:cdata(), 
     self.lambda,
     self.output:cdata()
   )
   return self.output
end

function HardShrink:updateGradInput(input, gradOutput)
   THNN.errcheck(
     'THNN_RealHardShrink_updateGradInput', 
     input:type(), 
     THNN.NULL, 
     input:cdata(), 
     self.lambda,
     gradOutput:cdata(),
     self.gradInput:cdata()
   )
   return self.gradInput
end
