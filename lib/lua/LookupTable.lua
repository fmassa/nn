local LookupTable, parent = torch.class('THNN.LookupTable', 'nn.Module', THNN)

LookupTable.__version = 4

function LookupTable:__init(nIndex, nOutput)
   parent.__init(self)

   self.weight = torch.Tensor(nIndex, nOutput)
   self.gradWeight = torch.Tensor(nIndex, nOutput):zero()

   self:reset()
end

function LookupTable:backCompatibility()
   self._input = self._input or torch.LongTensor()
   
   if not self._count then
      self._count = torch.LongTensor()
   elseif self._count and torch.isTypeOf(self._count, torch.IntTensor) then
      self._count = self._count:long()
   end
   
   if not self.shouldScaleGradByFreq then
      self.shouldScaleGradByFreq = false
   end
end

function LookupTable:accUpdateOnly()
   self.gradWeight = nil
   return self
end

function LookupTable:scaleGradByFreq()
   self.shouldScaleGradByFreq = true
   return self
end

function LookupTable:reset(stdv)
   stdv = stdv or 1
   self.weight:normal(0, stdv)
end

function LookupTable:makeInputContiguous(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._input) then
      self.copiedInput = true
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   self.copiedInput = false
   return input
end

function LookupTable:updateOutput(input)
   self:backCompatibility()
   input = self:makeInputContiguous(input)
   if input:dim() == 1 then
      self.output:index(self.weight, 1, input)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
   else
      error("input must be a vector or matrix")
   end
   return self.output
end

function LookupTable:accGradParameters(input, gradOutput, scale)
   self:backCompatibility()
   input = self.copiedInput and self._input or input
   if input:dim() == 2 then
      input = input:view(-1)
   elseif input:dim() ~= 1 then
      error("input must be a vector or matrix")
   end

   THNN.errcheck(
     'THNN_RealLookupTable_accGradParameters',
     gradOutput:type(),   -- we cannot use input here since it may be a LongTensor (CPU version)
     THNN.NULL,
     input:cdata(),
     gradOutput:cdata(),
     self.gradWeight:cdata(),
     scale or 1,
     self.shouldScaleGradByFreq or false,
     self._count:cdata()
   )
end

function LookupTable:type(type, tensorCache)
   parent.type(self, type, tensorCache)

   if type == 'torch.CudaTensor' then
      -- CUDA uses _sorted and _indices temporary tensors
      self._sorted = self.weight.new()
      self._indices = self.weight.new()
      self._count = self.weight.new()
      self._input = self.weight.new()
   else
      -- self._count and self._input should only be converted if using Cuda
      self._count = torch.LongTensor()
      self._input = torch.LongTensor()
   end

   return self
end

-- we do not need to accumulate parameters when sharing
LookupTable.sharedAccUpdateGradParameters = LookupTable.accUpdateGradParameters
