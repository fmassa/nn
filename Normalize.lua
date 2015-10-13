local Normalize, parent = torch.class('nn.Normalize', 'nn.Module')

function Normalize:__init(p,eps)
  parent.__init(self)
  assert(p,'p-norm not provided')
  assert(p > 0, p..'-norm not supported')
  self.p = p
  self.eps = eps or 1e-10
end

local function updateOutputInf(self,input)
  self._indices = self._indices or
      (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())

  torch.max(self.norm, self._indices, input, 2)
end

local function updateOutputLp(self,input)
  self.normp = self.normp or input.new()
  if self.p % 2 ~= 0 then
    self.buffer:abs(input):pow(self.p)
  else
    self.buffer:pow(input,self.p)
  end
  self.normp:sum(self.buffer,2):add(self.eps)
  self.norm:pow(self.normp,1/self.p)
end

function Normalize:updateOutput(input)
  assert(input:dim() <= 2, 'only 1d layer supported')
  local is_batch = true
  if input:dim() == 1 then
    input = input:view(1,-1)
    is_batch = false
  end

  self.output:resizeAs(input)

  self.norm = self.norm or input.new()
  self.buffer = self.buffer or input.new()

  if self.p == math.huge then
    updateOutputInf(self,input)
  else
    updateOutputLp(self,input)
  end

  self.output:cdiv(input,self.norm:view(-1,1):expandAs(self.output))

  if not is_batch then
    self.output = self.output[1]
  end
  return self.output
end

function Normalize:updateGradInput(input, gradOutput)
  assert(input:dim() <= 2, 'only 1d layer supported')
  assert(gradOutput:dim() <= 2, 'only 1d layer supported')

  local is_batch = true
  if input:dim() == 1 then
    input = input:view(1,-1)
    is_batch = false
  end

  local n = input:size(1) -- batch size
  local d = input:size(2) -- dimensionality of vectors

  -- compute diagonal term with gradOutput
  self.gradInput:resize(n,d,1)
  gradOutput = gradOutput:view(n,d,1)

  if self.p == math.huge then
    self.gradInput:cmul(self.norm:view(n,1,1):expand(n,d,1),gradOutput)
    self.buffer:resizeAs(input):zero()
    self.buffer:scatter(2,self._indices,1)
  else
    self.gradInput:cmul(self.normp:view(n,1,1):expand(n,d,1),gradOutput)
    self.buffer:abs(input):pow(self.p-2):cmul(input)
  end
  -- compute cross term in two steps
  self.cross = self.cross or input.new()
  self.cross:resize(n,1,1)

  local b1 = self.buffer:view(n,d,1)
  local b2 = input:view(n,1,d)
  -- instead of having a huge temporary matrix (b1*b2),
  -- do the computations as b1*(b2*gradOutput). This avoids redundant
  -- computation and also a huge buffer of size n*d^2
  self.cross:bmm(b2,gradOutput)
  self.gradInput:baddbmm(-1,b1, self.cross)

  -- reuse cross buffer for normalization
  if self.p == math.huge then
    self.cross:cmul(self.norm,self.norm)
  else
    self.cross:cmul(self.normp,self.norm)
  end
  self.gradInput:cdiv(self.cross:view(n,1,1):expand(n,d,1))

  self.gradInput = self.gradInput:view(n,d)
  
  if not is_batch then
    self.gradInput = self.gradInput[1]
  end

  return self.gradInput
end

function Normalize:__tostring__()
  local s
  -- different prints if the norm is integer
  if self.p % 1 == 0 then
    s = '%s(%d)'
  else
    s = '%s(%f)'
  end
  return string.format(s,torch.type(self),self.p)
end

function Normalize:type(type)
  -- torch.max expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
  if type == 'torch.CudaTensor' then
    parent.type(self, type)
  else
    -- self._indices must be a LongTensor. Setting it to nil temporarily avoids
    -- unnecessary memory allocations.
    local indices
    indices, self._indices = self._indices, nil
    parent.type(self, type)
    self._indices = indices and indices:long() or nil
  end
  return self
end
