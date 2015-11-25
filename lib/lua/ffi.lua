local ffi = require "ffi"

local base_str = [[
typedef void THNNState;
void THNN_TYPESpatialConvolution_updateOutput(THNNState* state,
                                           THTensor* input,
                                           THTensor* output,
                                           THTensor* weight,
                                           THTensor* bias,
                                           THTensor* finput,
                                           int kW,   int kH,
                                           int dW,   int dH,
                                           int padW, int padH);

void THNN_TYPESpatialConvolution_updateGradInput(THNNState* state,
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

void THNN_TYPESpatialConvolution_accGradParameters(THNNState* state,
                                           THTensor* input,
                                           THTensor* gradOutput,
                                           THTensor* gradWeight,
                                           THTensor* gradBias,
                                           THTensor* finput,
                                           real scale);

void THNN_TYPEAbs_updateOutput(THNNState* state, THTensor *input, THTensor *output);
void THNN_TYPEAbs_updateGradInput(THNNState* state, THTensor *input, THTensor *gradOutput, THTensor *gradInput);

void THNN_TYPEAbsCriterion_updateOutput(THNNState* state, THTensor *input, THTensor *target, real *output, bool sizeAverage);
void THNN_TYPEAbsCriterion_updateGradInput(THNNState* state, THTensor *input, THTensor *target, THTensor *gradInput, bool sizeAverage);

void THNN_TYPEClassNLLCriterion_updateOutput(THNNState* state, THTensor *input, THLongTensor *target, THTensor *output, bool sizeAverage, THTensor *weights, THTensor *total_weight);
void THNN_TYPEClassNLLCriterion_updateGradInput(THNNState* state, THTensor *input, THLongTensor *target, THTensor *gradInput, bool sizeAverage, THTensor *weights, THTensor *total_weight);

void THNN_TYPEDistKLDivCriterion_updateOutput(THNNState* state, THTensor *input, THTensor *target, real *output, bool sizeAverage);
void THNN_TYPEDistKLDivCriterion_updateGradInput(THNNState* state, THTensor *input, THTensor *target, THTensor *gradInput, bool sizeAverage);

void THNN_TYPEHardShrink_updateOutput(THNNState* state, THTensor *input, THTensor *output, real lambda);
void THNN_TYPEHardShrink_updateGradInput(THNNState* state, THTensor *input, THTensor *gradOutput, THTensor *gradInput, real lambda);
]]

local temp_str = {}

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

