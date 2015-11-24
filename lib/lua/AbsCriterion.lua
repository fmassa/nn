local AbsCriterion, parent = torch.class('THNN.AbsCriterion', 'nn.Criterion', THNN)

function AbsCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function AbsCriterion:updateOutput(input, target)
   self.output_tensor =  self.output_tensor or input.new(1)
   THNN.errcheck(
      'THNN_RealAbsCriterion_updateOutput', 
      input:type(), 
      THNN.NULL, 
      input:cdata(), 
      target:cdata(),
      self.sizeAverage,
      self.output_tensor:data()
   )
   self.output = self.output_tensor[1]
   return self.output
end

function AbsCriterion:updateGradInput(input, target)
   THNN.errcheck(
      'THNN_RealAbsCriterion_updateOutput', 
      input:type(), 
      THNN.NULL, 
      input:cdata(), 
      target:cdata(),
      self.sizeAverage,
      self.gradInput:cdata()
   )
   return self.gradInput
end
