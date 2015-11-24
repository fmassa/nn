local Abs, parent = torch.class('THNN.Abs', 'nn.Module', THNN)

function Abs:__init()
   parent.__init(self)
end

function Abs:updateOutput(input)
   THNN.errcheck(
     'THNN_RealAbs_updateOutput', 
     input:type(), 
     THNN.NULL, 
     input:cdata(), 
     self.output:cdata()
   )
   return self.output
end

function Abs:updateGradInput(input, gradOutput)
   THNN.errcheck(
     'THNN_RealAbs_updateGradInput', 
     input:type(), 
     THNN.NULL, 
     input:cdata(), 
     gradOutput:cdata(),
     self.gradInput:cdata()
   )
   return self.gradInput
end
