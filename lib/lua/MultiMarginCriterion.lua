local MultiMarginCriterion, parent = torch.class('THNN.MultiMarginCriterion', 'nn.Criterion', THNN)

function MultiMarginCriterion:__init(p)
   assert(p == nil or p == 1 or p == 2, 'only p=1 and p=2 supported')
   self.p = p or 1
   parent.__init(self)
   self.sizeAverage = true
end

function MultiMarginCriterion:updateOutput(input, target)
   -- backward compatibility
   local _target = target
   if not torch.isTensor(_target) then
     _target = input.new(1)
     _target[1] = target
   end
   self.p = self.p or 1
   self.output_tensor = self.output_tensor or input.new(1)
   THNN.errcheck(
      'THNN_RealMultiMarginCriterion_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      _target:cdata(),
      self.output_tensor:data(),
      self.sizeAverage,
      self.p
   )
   self.output = self.output_tensor[1]
   return self.output
end

function MultiMarginCriterion:updateGradInput(input, target)
   local _target = target
   if not torch.isTensor(_target) then
     _target = input.new(1)
     _target[1] = target
   end
   THNN.errcheck(
      'THNN_RealMultiMarginCriterion_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      _target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage,
      self.p
   )
   return self.gradInput
end
