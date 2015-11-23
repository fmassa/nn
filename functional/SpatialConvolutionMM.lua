-- handle NULL in a better manner
local ffi = require "ffi"
ffi.NULL = ffi.NULL or nil

-- these functions require weight and bias to be initialized
-- should probably create another function like createWeightDescriptors
--
-- only handles 4D data
function SpatialConvolution_updateOutput(input,weight,bias,finput,output,kW,kH,dW,dH,padW,padH)
  THNN.errcheck('THNN_RealSpatialConvolution_updateOutput',input:type(),ffi.NULL,input:cdata(),weight:cdata(),bias:cdata(),finput:cdata(),output:cdata(),kW,kH,dW,dH,padW,padH)
end

function SpatialConvolution_updateGradInput(input,weight,bias,gradOutput,finput,fgradInput,gradInput,kW,kH,dW,dH,padW,padH)
  THNN.errcheck('THNN_RealSpatialConvolution_updateGradInput',input:type(), ffi.NULL,input:cdata(),weight:cdata(),bias:cdata(),gradOutput:cdata(),finput:cdata(),fgradInput:cdata(),gradInput:cdata(),kW,kH,dW,dH,padW,padH)
end

function SpatialConvolution_accGradParameters(input,gradWeight,gradBias,gradOutput,finput,scale)
  THNN.errcheck('THNN_RealSpatialConvolution_accGradParameters',input:type(), ffi.NULL,input:cdata(),gradWeight:cdata(),gradBias:cdata(),gradOutput:cdata(),finput:cdata(),scale)
end
