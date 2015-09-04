local Contiguous, parent = torch.class('nn.Contiguous', 'nn.Module')

function Contiguous:__init()
   parent.__init(self)
end

function Contiguous:updateOutput(input)
   if not input:isContiguous() then
      self.buffer = self.buffer or input.new()
      self.buffer:resizeAs(input)
      self.buffer:copy(input)
      input = self.buffer
   end
   self.output = input
   return self.output
end

function Contiguous:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self.buffer = self.buffer or input.new()
      self.buffer:resizeAs(gradOutput)
      self.buffer:copy(gradOutput)
      gradOutput = self.buffer
   end
   self.gradInput = gradOutput
   return self.gradInput
end
