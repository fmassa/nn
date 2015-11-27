local SmoothL1Criterion, parent = torch.class('THNN.SmoothL1Criterion', 'nn.Criterion', THNN)

function SmoothL1Criterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function SmoothL1Criterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   THNN.errcheck(
      'THNN_RealSmoothL1Criterion_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      target:cdata(),
      self.output_tensor:data(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   return self.output
end

function SmoothL1Criterion:updateGradInput(input, target)
   THNN.errcheck(
      'THNN_RealSmoothL1Criterion_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
   return self.gradInput
end
