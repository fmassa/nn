local ffi = require "ffi"

local base_str = [[
typedef void THNNState;

TH_API void THNN_(Abs_updateOutput)(
          THNNState* state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Abs_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(AbsCriterion_updateOutput)(
          THNNState* state,
          THTensor *input,
          THTensor *target,
          real *output,
          bool sizeAverage);
TH_API void THNN_(AbsCriterion_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(ClassNLLCriterion_updateOutput)(
          THNNState* state,
          THTensor *input,
          THLongTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);
TH_API void THNN_(ClassNLLCriterion_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THLongTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);

TH_API void THNN_(DistKLDivCriterion_updateOutput)(
          THNNState* state,
          THTensor *input,
          THTensor *target,
          real *output,
          bool sizeAverage);
TH_API void THNN_(DistKLDivCriterion_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(HardShrink_updateOutput)(
          THNNState* state,
          THTensor *input,
          THTensor *output,
          real lambda);
TH_API void THNN_(HardShrink_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real lambda);

TH_API void THNN_(HardTanh_updateOutput)(
          THNNState* state,
          THTensor *input,
          THTensor *output,
          real min_val,
          real max_val);
TH_API void THNN_(HardTanh_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real min_val,
          real max_val);

TH_API void THNN_(L1Cost_updateOutput)(
          THNNState* state,
          THTensor *input,
          real *output);
TH_API void THNN_(L1Cost_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *gradInput);
          
TH_API void THNN_(SpatialConvolution_updateOutput)(
          THNNState* state,
          THTensor* input,
          THTensor* output,
          THTensor* weight,
          THTensor* bias,
          THTensor* finput,
          int kW,   int kH,
          int dW,   int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolution_updateGradInput)(
          THNNState* state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradInput,
          THTensor* weight,
          THTensor* bias,
          THTensor* finput,
          THTensor* fgradInput,
          int kW,   int kH,
          int dW,   int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolution_accGradParameters)(
          THNNState* state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradWeight,
          THTensor* gradBias,
          THTensor* finput,
          real scale);

TH_API void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState* state,
          THTensor* input,
          THTensor* output,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int ceil_mode,
          THTensor* indices);
TH_API void THNN_(SpatialMaxPooling_updateGradInput)(
          THNNState* state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradInput,
          int dW, int dH,
          THTensor* indices);

TH_API void THNN_(Threshold_updateOutput)(
          THNNState* state,
          THTensor* input,
          THTensor* output,
          real val,
          real threshold,
          int inPlace);

TH_API void THNN_(Threshold_updateGradInput)(
          THNNState* state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradInput,
          real threshold,
          int inPlace);
]]

local temp_str = {}

base_str = string.gsub(base_str, "TH_API void THNN_%(([%a%d_]+)%)", 'void THNN_TYPE%1') -- conversion used for generic/THNN.h

temp_str[1] = string.gsub(base_str,'TYPE','Double')
temp_str[1] = string.gsub(temp_str[1],'real','double')
temp_str[1] = string.gsub(temp_str[1],'THTensor','THDoubleTensor')

temp_str[2] = string.gsub(base_str,'TYPE','Float')
temp_str[2] = string.gsub(temp_str[2],'real','float')
temp_str[2] = string.gsub(temp_str[2],'THTensor','THFloatTensor')

ffi.cdef(table.concat(temp_str))

local ok,err
if ffi.os == "OSX" then
  ok,err = pcall(function() THNN.C = ffi.load('../THNN/libTHNN.dylib') end) -- fix path later
else
  ok,err = pcall(function() THNN.C = ffi.load('../THNN/libTHNN.so') end) -- fix path later
end
if not ok then
  print(err)
  error('Ops')
end
--local C = ffi.load(paths.cwd() .. '/libTHNN.so')

