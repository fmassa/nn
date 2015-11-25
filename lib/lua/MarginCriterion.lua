local MarginCriterion, parent = torch.class('THNN.MarginCriterion', 'nn.Criterion', THNN)

function MarginCriterion:__init(margin)
   parent.__init(self)
   self.sizeAverage = true
   self.margin = margin or 1
end

function MarginCriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   THNN.errcheck(
      'THNN_RealMarginCriterion_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      target:cdata(),
      self.output_tensor:data(),
      self.margin,
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   return self.output
end

function MarginCriterion:updateGradInput(input, target)
   THNN.errcheck(
      'THNN_RealMarginCriterion_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.margin,
      self.sizeAverage
   )
   return self.gradInput
end
