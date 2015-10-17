local SpatialCrossLRN, parent = torch.class('nn.SpatialCrossLRN', 'nn.Module')

function SpatialCrossLRN:__init(size, alpha, beta, k)
  parent.__init(self)

  self.size = size
  self.alpha = alpha or 0.0001
  self.beta = beta or 0.75
  self.k = k or 1

  self.scale = torch.Tensor()

end

local function createMatrix(self,numFilters)
  local mat = torch.Tensor(numFilters, numFilters):zero()
  local idx = torch.range(1-(self.size-1)/2,numFilters+(self.size-1)/2):long()
  idx:clamp(1,numFilters)
  idx = idx:unfold(1,self.size,1)
  mat:scatter(1,idx:t(),1)
  return mat
end

function SpatialCrossLRN:updateOutputFast(input)
  assert(input:dim() == 3 or input:dim() == 4,
         'Input must be 3D or 4D')
  local isBatch = true
  if input:dim() == 3 then
    input = nn.utils.addSingletonDimension(input)
    isBatch = false
  end

  local tt = torch.Timer()
  local batchSize   = input:size(1)
  local channels    = input:size(2) 
  local inputHeight = input:size(3) 
  local inputWidth  = input:size(4) 

  self.output:resizeAs(input)

tt:reset()
  -- use output storage as temporary buffer
  local inputSquare = self.output
  inputSquare:pow(input:transpose(2,3):transpose(3,4), 2):resize(input:size(1)*input:size(3)*input:size(4),input:size(2))
print('input square non cont',tt:time().real)
tt:reset()
  self.cmat = createMatrix(self,input:size(2))

print('create matrix', tt:time().real)

  --self.scale:resizeAs(input)
  self.scale:resizeAs(inputSquare):fill(self.k)

tt:reset()
  self.scale:addmm(self.alpha/self.size,inputSquare,self.cmat)

print('addmm',tt:time().real)

  --self.scale:mul(self.alpha/self.size):add(self.k)

tt:reset()
local cc = self.scale:view(input:size(1),input:size(3),input:size(4),input:size(2)):transpose(2,4):transpose(3,4):clone()

print('view+clone',tt:time().real)
tt:reset()
  self.output:pow(cc,-self.beta)

print('pow -beta',tt:time().real)
tt:reset()
  self.output:cmul(input)
print('cmul',tt:time().real)

  if not isBatch then
    self.output = self.output[1]
  end

  return self.output
end



function SpatialCrossLRN:updateOutput(input)
  assert(input:dim() == 3 or input:dim() == 4,
         'Input must be 3D or 4D')
  local isBatch = true
  if input:dim() == 3 then
    input = nn.utils.addSingletonDimension(input)
    isBatch = false
  end

  local batchSize   = input:size(1)
  local channels    = input:size(2) 
  local inputHeight = input:size(3) 
  local inputWidth  = input:size(4) 

  self.output:resizeAs(input)
  self.scale:resizeAs(input)

  -- use output storage as temporary buffer
  local inputSquare = self.output
  inputSquare:pow(input, 2)
    
  local prePad = (self.size - 1)/2 + 1
  local prePadCrop = prePad > channels and channels or prePad

  local scaleFirst = self.scale:select(2,1)
  scaleFirst:zero()
  -- compute first feature map normalization
  for c = 1, prePadCrop do
    scaleFirst:add(inputSquare:select(2, c))
  end

  -- reuse computations for next feature maps normalization
  -- by adding the next feature map and removing the previous
  for c = 2, channels do
    local scalePrevious = self.scale:select(2, c -1)
    local scaleCurrent  = self.scale:select(2, c)
    scaleCurrent:copy(scalePrevious)
    if c < channels - prePad + 2 then
      local squareNext   = inputSquare:select(2, c + prePad - 1)
      scaleCurrent:add(1, squareNext)
    end
    if c > prePad  then
      local squarePrevious = inputSquare:select(2, c - prePad )
      scaleCurrent:add(-1, squarePrevious)
    end
  end

  self.scale:mul(self.alpha/self.size):add(self.k)

  self.output:pow(self.scale,-self.beta)
  self.output:cmul(input)

  if not isBatch then
    self.output = self.output[1]
  end

  return self.output
end

function SpatialCrossLRN:updateGradInput(input, gradOutput)
  assert(input:dim() == 3 or input:dim() == 4,
         'Input must be 3D or 4D')
  local isBatch = true
  if input:dim() == 3 then
    input = nn.utils.addSingletonDimension(input)
    gradOutput = nn.utils.addSingletonDimension(gradOutput)
    self.output = nn.utils.addSingletonDimension(self.output)
    isBatch = false
  end

  local batchSize   = input:size(1)
  local channels    = input:size(2) 
  local inputHeight = input:size(3) 
  local inputWidth  = input:size(4) 

  self.paddedRatio = self.paddedRatio or input.new()
  self.accumRatio = self.accumRatio or input.new()
  self.paddedRatio:resize(channels + self.size - 1, inputHeight, inputWidth)
  self.accumRatio:resize(inputHeight,inputWidth)

  local cacheRatioValue = 2*self.alpha*self.beta/self.size
  local inversePrePad = self.size - (self.size - 1) / 2

  self.gradInput:resizeAs(input)
  self.gradInput:pow(self.scale,-self.beta):cmul(gradOutput)

  self.paddedRatio:zero()
  local paddedRatioCenter = self.paddedRatio:narrow(1, inversePrePad, channels)
  for n = 1, batchSize do
    paddedRatioCenter:cmul(gradOutput[n],self.output[n])
    paddedRatioCenter:cdiv(self.scale[n])
    self.accumRatio:sum(self.paddedRatio:narrow(1,1,self.size-1), 1)
    for c = 1, channels do
      self.accumRatio:add(self.paddedRatio[c+self.size-1])
      self.gradInput[n][c]:addcmul(-cacheRatioValue, input[n][c], self.accumRatio)
      self.accumRatio:add(-1, self.paddedRatio[c])
    end
  end

  if not isBatch then
    self.gradInput = self.gradInput[1]
    self.output = self.output[1]
  end

  return self.gradInput
end
