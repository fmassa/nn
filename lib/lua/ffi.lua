local ffi = require "ffi"

local base_str = [[
typedef void THNNState;
void THNN_TYPESpatialConvolution_updateOutput(THNNState* state,
                                           THTensor* input,
                                           THTensor* weight,
                                           THTensor* bias,
                                           THTensor* finput,
                                           THTensor* output,
                                           int kW,   int kH,
                                           int dW,   int dH,
                                           int padW, int padH);

void THNN_TYPESpatialConvolution_updateGradInput(THNNState* state,
                                           THTensor* input,
                                           THTensor* weight,
                                           THTensor* bias,
                                           THTensor* gradOutput,
                                           THTensor* finput,
                                           THTensor* fgradInput,
                                           THTensor* gradInput,
                                           int kW,   int kH,
                                           int dW,   int dH,
                                           int padW, int padH);

void THNN_TYPESpatialConvolution_accGradParameters(THNNState* state,
                                           THTensor* input,
                                           THTensor* gradWeight,
                                           THTensor* gradBias,
                                           THTensor* gradOutput,
                                           THTensor* finput,
                                           real scale);

int THNN_TYPEAbs_updateOutput(THNNState* state, THTensor *input, THTensor *output);
int THNN_TYPEAbs_updateGradInput(THNNState* state, THTensor *input, THTensor *gradOutput, THTensor *gradInput);

int THNN_TYPEAbsCriterion_updateOutput(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, real *output);
int THNN_TYPEAbsCriterion_updateGradInput(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, THTensor *gradInput);

int THNN_TYPEClassNLLCriterion_updateOutput(THNNState* state, THTensor *input, THLongTensor *target, bool sizeAverage, THTensor *weights, THTensor *total_weight, THTensor *output);
int THNN_TYPEClassNLLCriterion_updateGradInput(THNNState* state, THTensor *input, THLongTensor *target, bool sizeAverage, THTensor *weights, THTensor *total_weight, THTensor *gradInput);

]]

local temp_str = {}

temp_str[1] = string.gsub(base_str,'TYPE','Double')
temp_str[1] = string.gsub(temp_str[1],'real','double')
temp_str[1] = string.gsub(temp_str[1],'THTensor','THDoubleTensor')

temp_str[2] = string.gsub(base_str,'TYPE','Float')
temp_str[2] = string.gsub(temp_str[2],'real','float')
temp_str[2] = string.gsub(temp_str[2],'THTensor','THFloatTensor')

ffi.cdef(table.concat(temp_str))


local ok,err = pcall(function() THNN.C = ffi.load('../THNN/libTHNN.so') end) -- fix path later
if not ok then
  print(err)
  error('Ops')
end
--local C = ffi.load(paths.cwd() .. '/libTHNN.so')

