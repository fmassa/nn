local L1Cost, parent = torch.class('THNN.L1Cost','nn.Criterion', THNN)

function L1Cost:__init()
   parent.__init(self)
end

function L1Cost:updateOutput(input)
   self.output_tensor = self.output_tensor or input.new(1)
   THNN.errcheck(
      'THNN_RealL1Cost_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.output_tensor:data()
   )
   self.output = self.output_tensor[1]
   return self.output
end

function L1Cost:updateGradInput(input)
   THNN.errcheck(
      'THNN_RealL1Cost_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.gradInput:cdata()
   )
   return self.gradInput
end

