local MSECriterion, parent = torch.class('THNN.MSECriterion', 'nn.Criterion', THNN)

function MSECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MSECriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   THNN.errcheck(
      'THNN_RealMSECriterion_updateOutput',
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
          
function MSECriterion:updateGradInput(input, target)
   THNN.errcheck(
      'THNN_RealMSECriterion_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
   return self.gradInput
end
