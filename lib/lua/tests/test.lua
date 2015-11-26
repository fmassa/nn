-- this file need to be runned for the moment
-- in the previous folder
dofile 'init.lua'
require 'nn'

local mytester = torch.Tester()
local jac

local precision = 1e-5
local high_precision = 1e-12

local nntest = {}

local function criterionJacobianTest1D(cri, input, target)
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end



function nntest.Abs()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = THNN.Abs()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- compare against nn
   local nnmodule = nn.Abs()
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end

function nntest.ClassNLLCriterion()
   local numLabels = math.random(5,10)
   local input = torch.rand(numLabels)
   local target = math.random(1,numLabels)

   -- default ClassNLLCriterion
   local cri = THNN.ClassNLLCriterion()
   criterionJacobianTest1D(cri, input, target)

   -- ClassNLLCriterion with weights
   local weights = torch.rand(numLabels)
   weights = weights / weights:sum()
   cri = THNN.ClassNLLCriterion(weights)
   criterionJacobianTest1D(cri, input, target)

   -- compare against nn
   local cri = THNN.ClassNLLCriterion()
   local nncri = nn.ClassNLLCriterion()
   local output = cri:forward(input,target)
   local output_nn = nncri:forward(input,target)
   local err = math.abs(output-output_nn)

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')

   local cri = THNN.ClassNLLCriterion(weights)
   local nncri = nn.ClassNLLCriterion(weights)
   local output = cri:forward(input,target)
   local output_nn = nncri:forward(input,target)
   local err = math.abs(output-output_nn)

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')

end

function nntest.DistKLDivCriterion()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = THNN.DistKLDivCriterion(true)  -- sizeAverage = true
   criterionJacobianTest1D(cri, input, target)
   cri = THNN.DistKLDivCriterion(false)  -- sizeAverage = false
   criterionJacobianTest1D(cri, input, target)
   
   -- compare against nn
   local cri = THNN.DistKLDivCriterion()
   local nncri = nn.DistKLDivCriterion()
   local output = cri:forward(input,target)
   local output_nn = nncri:forward(input,target)
   local err = math.abs(output-output_nn)

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end

function nntest.HardShrink()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local lambda = math.random()/2
   local module = THNN.HardShrink(lambda)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   
   -- compare against nn
   local nnmodule = nn.HardShrink(lambda)
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end

function nntest.HardTanh()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = THNN.HardTanh()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   
   -- compare against nn
   local nnmodule = nn.HardTanh()
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end

function nntest.L1Cost()
   local ini = math.random(3,5)
   local input = torch.rand(ini)

   local module = THNN.L1Cost()

   -- compare against nn
   local nnmodule = nn.L1Cost()
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = math.abs(output-output_nn)

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end

function nntest.LogSigmoid()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = THNN.LogSigmoid()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   
   -- compare against nn
   local nnmodule = nn.LogSigmoid()
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end

function nntest.LogSoftmax()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local input = torch.Tensor(ini,inj):zero()
   local module = THNN.LogSoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err, 1e-3, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- compare against nn
   local nnmodule = nn.LogSoftMax()
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end

function nntest.LookupTable()
   local totalIndex = math.random(6,9)
   local nIndex = math.random(3,5)
   local entry_size = math.random(2,5)
   local input = torch.randperm(totalIndex):narrow(1,1,nIndex):int()
   local module = THNN.LookupTable(totalIndex, entry_size)
   local minval = 1
   local maxval = totalIndex

   local output = module:forward(input)
   module:backwardUpdate(input, output, 0.1)
   input:zero()

   -- 1D
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
   mytester:assertlt(err,precision, '1D error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
   mytester:assertlt(err,precision, '1D error on weight [direct update] ')

   module.gradWeight:zero()
   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         '1D error on weight [%s]', t))
   end

   -- 2D
   local nframe = math.random(2,5)
   local input = torch.IntTensor(nframe, nIndex):zero()

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
   mytester:assertlt(err,precision, '2D error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
   mytester:assertlt(err,precision, '2D error on weight [direct update] ')

   module.gradWeight:zero()
   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         '2D error on weight [%s]', t))
   end

   -- IO
   module.gradInput = torch.Tensor(3,4):zero() --fixes an error
   local ferr,berr = jac.testIO(module,input,minval,maxval)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- accUpdate
   module:accUpdateOnly()
   mytester:assert(not module.gradWeight, 'gradWeight is nil')
   module:float()
   local output = module:forward(input)
   module:backwardUpdate(input, output, 0.1)
   
   -- compare against nn
   local nnmodule = nn.LookupTable(totalIndex, entry_size):float()
   nnmodule.weight = module.weight
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision, 'error comparing against nn ')
end

function nntest.MarginCriterion()
   local input = torch.rand(100)
   local target = input:clone():add(torch.rand(100))
   local cri = THNN.MarginCriterion()
   criterionJacobianTest1D(cri, input, target)
   
   -- compare against nn
   local nncri = nn.MarginCriterion()
   local output = cri:forward(input,target)
   local output_nn = nncri:forward(input,target)
   local err = math.abs(output-output_nn)

   mytester:assertlt(err, high_precision, 'error comparing against nn ')
end

function nntest.MSECriterion()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = THNN.MSECriterion()
   criterionJacobianTest1D(cri, input, target)
   
   -- compare against nn
   local nncri = nn.MSECriterion()
   local output = cri:forward(input,target)
   local output_nn = nncri:forward(input,target)
   local err = math.abs(output-output_nn)

   mytester:assertlt(err, high_precision, 'error comparing against nn ')
end

function nntest.MultiLabelMarginCriterion()
   local input = torch.randn(2, 4)
   local target = torch.Tensor{{1, 3, 0, 0}, {4, 0, 0, 0}} -- zero-values are ignored
   local cri = THNN.MultiLabelMarginCriterion()
   criterionJacobianTest1D(cri, input, target)

   -- compare against nn
   local nncri = nn.MultiLabelMarginCriterion()
   local output = cri:forward(input,target)
   local output_nn = nncri:forward(input,target)
   local err = math.abs(output-output_nn)

   mytester:assertlt(err, high_precision, 'error comparing against nn ')
end

function nntest.MultiMarginCriterion()
   local input = torch.rand(100)
   local target = math.random(1,100)
   local p = math.random(1,2)
   local cri = THNN.MultiMarginCriterion(p)
   criterionJacobianTest1D(cri, input, target)
   
   -- compare against nn
   local nncri = nn.MultiMarginCriterion(p)
   local output = cri:forward(input,target)
   local output_nn = nncri:forward(input,target)
   local err = math.abs(output-output_nn)

   mytester:assertlt(err, high_precision, 'error comparing against nn ')
end

function nntest.PReLU()
   local ini = math.random(3,5)
   local input = torch.Tensor(ini):zero()

   local module = THNN.PReLU(ini)

   -- 1D
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                        'error on weight [%s]', t))
   end

   -- 2D
   local nframe = math.random(1,7)
   local input = torch.Tensor(nframe, ini):zero()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                        'error on weight [%s]', t))
   end

   -- 4D
   local nframe = math.random(1,7)
   local kW, kH = math.random(1,8), math.random(1,8)
   local input = torch.Tensor(nframe, ini, kW, kH):zero()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                        'error on weight [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   
   -- compare against nn
   local nnmodule = nn.PReLU(ini)
   nnmodule.weight = module.weight
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision, 'error comparing against nn ')
end

function nntest.RReLU()
   local nframe = math.random(1,7)
   local size = math.random(1,7)
   local kW, kH = math.random(1,8), math.random(1,8)
   local input = torch.Tensor(nframe, size, kW, kH):zero()

   local l = 1/math.random(5,8)
   local u = 1/math.random(3,5)

   -- test in evaluation mode (not inplace), RReLU behaves like LeakyReLU
   local module = THNN.RReLU(l, u, false)
   mytester:assert(module.train, 'default mode ')
   module:evaluate()

   -- gradient check
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
 
   -- IO
   local ferr,berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   
   -- test training and evalation mode
   for _,train in ipairs({true,false}) do
      -- test with separate output buffer and inplace
      for _,inplace in ipairs({false,true}) do
         module = THNN.RReLU(l, u, inplace)
         if train then
            module:training()
         else
            module:evaluate()
         end
         input = torch.rand(nframe, size, kW, kH) - 0.5
         input:storage()[1] = -1
         local original_input = input:clone()
         local output = module:forward(input)
         mytester:assert(output:sign():eq(original_input:sign()):all(), 'sign flipped forward ')
         local gradOutput = torch.ones(output:size())
         local gradInput = module:backward(input, gradOutput)
         mytester:assert(gradInput:gt(0):eq(input:ne(0)):all(), 'gradient ')
         mytester:assert(gradInput:lt(1):eq(input:le(0)):all(), 'backward negative inputs ')
         mytester:assert(gradInput:eq(1):eq(input:gt(0)):all(), 'backward positive inputs ') 
         if not train then
            local err = gradInput[input:le(0)]:mean()-(module.lower+module.upper)/2
            mytester:assertlt(err, precision, 'error on gradient ')
         end

         input = -torch.rand(1000)
         module:forward(input) -- fill internal noise tensor
         local g = module:backward(input, torch.ones(1000))
         local err = math.abs(g[input:le(0)]:mean()-(module.lower+module.upper)/2)
         mytester:assertlt(err, 0.05, 'mean deviation of gradient for negative inputs ')
      end
   end
end

function nntest.LogSigmoid()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = THNN.LogSigmoid()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   
   -- compare against nn
   local nnmodule = nn.LogSigmoid()
   nnmodule.weight = module.weight
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision, 'error comparing against nn ')
end

function nntest.SpatialConvolutionMM()
   local from = math.random(2,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local di =  math.random(1,4)
   local dj =  math.random(1,4)
   local padW = math.random(0,2)
   local padH = math.random(0,2)
   local outi = math.random(5,9)
   local outj = math.random(5,9)
   local ini = (outi-1)*di+ki-padW*2
   local inj = (outj-1)*dj+kj-padH*2
   local module = THNN.SpatialConvolutionMM(from, to, ki, kj, di, dj, padW, padH)
   local input = torch.Tensor(from, inj, ini):zero()

   -- stochastic

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- batch

   --verbose = true
   local batch = math.random(2,5)

   module = THNN.SpatialConvolutionMM(from, to, ki, kj, di, dj, padW, padH)
   input = torch.Tensor(batch,from,inj,ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

   -- non-contiguous
   local input = torch.randn(batch,from,ini,inj):transpose(3,4) -- non-contiguous
   local inputc = input:contiguous() -- contiguous
   local output = module:forward(input):clone()
   local outputc = module:forward(inputc):clone()
   mytester:asserteq(0, (output-outputc):abs():max(), torch.typename(module) .. ' - contiguous err ')
   local gradInput = module:backward(input, output):clone()
   local gradInputc = module:backward(inputc, outputc):clone()
   mytester:asserteq(0, (gradInput-gradInputc):abs():max(), torch.typename(module) .. ' - contiguous err ')
end

function nntest.SpatialMaxPooling()
   for _,ceil_mode in pairs({true,false}) do
      local from = math.random(1,5)
      local ki = math.random(1,4)
      local kj = math.random(1,4)
      local si = math.random(1,3)
      local sj = math.random(1,3)
      local outi = math.random(4,5)
      local outj = math.random(4,5)
      local padW = math.min(math.random(0,1),math.floor(ki/2))
      local padH =  math.min(math.random(0,1),math.floor(kj/2))
      local ini = (outi-1)*si+ki-2*padW
      local inj = (outj-1)*sj+kj-2*padH

      local ceil_string = ceil_mode and 'ceil' or 'floor'
      local module = THNN.SpatialMaxPooling(ki,kj,si,sj,padW,padH)
      if ceil_mode then module:ceil() else module:floor() end
      local input = torch.rand(from,inj,ini)

      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error '..ceil_string..' mode on state ')

      local ferr, berr = jac.testIO(module, input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

      -- batch
      local nbatch = math.random(2,5)
      input = torch.rand(nbatch,from,inj,ini)
      module = THNN.SpatialMaxPooling(ki,kj,si,sj,padW,padH)
      if ceil_mode then module:ceil() else module:floor() end

      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error '..ceil_string..' mode on state (Batch)')

      local ferr, berr = jac.testIO(module, input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')
  end
end

function nntest.Sqrt()
   local in1 = torch.rand(5,7)
   local module = THNN.Sqrt()
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:assertlt(err, 1e-15, torch.typename(module) .. ' - forward err ')

   -- Test zero inputs; we will avoid a div-by-zero by setting to zero
   local zin = torch.DoubleTensor(5, 7):zero()
   module:forward(zin)
   local zgradout = torch.rand(5, 7)
   local zgradin = module:backward(zin, zgradout)
   mytester:assertTensorEq(zgradin, torch.DoubleTensor(5, 7):zero(), 0.000001, "error in sqrt backward singularity")

   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Sqrt()

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input, 0, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

    -- compare against nn
   local nnmodule = nn.Sqrt()
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end


function nntest.Threshold()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local th = torch.uniform(-2,2)
   local v = torch.uniform(-2,2)
   local module = THNN.Threshold(th,v)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

    -- compare against nn
   local nnmodule = nn.Threshold(th,v)
   local output = module:forward(input)
   local output_nn = nnmodule:forward(input)
   local err = (output-output_nn):abs():max()

   mytester:assertlt(err, high_precision ,  'error comparing against nn ')
end


mytester:add(nntest)
--if not nn then
if true then
   require 'nn'
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   mytester:run()
else
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   function nn.test(tests)
      -- randomize stuff
      math.randomseed(os.time())
      mytester:run(tests)
      return mytester
   end
end
