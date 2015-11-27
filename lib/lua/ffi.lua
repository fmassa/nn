local ffi = require "ffi"

local base_str = [[
typedef void THNNState;

// from torch7/lib/TH/THRandom.h
typedef struct {
  /* The initial seed. */
  unsigned long the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  unsigned long next;
  unsigned long state[624]; /* the array for the state vector 624 = _MERSENNE_STATE_N  */
  /********************************/
  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
} THGenerator;

TH_API void THNN_(Abs_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Abs_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(AbsCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          real *output,
          bool sizeAverage);
TH_API void THNN_(AbsCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(ClassNLLCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);
TH_API void THNN_(ClassNLLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);

TH_API void THNN_(DistKLDivCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          real *output,
          bool sizeAverage);
TH_API void THNN_(DistKLDivCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(HardShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real lambda);
TH_API void THNN_(HardShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real lambda);

TH_API void THNN_(HardTanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real min_val,
          real max_val);
TH_API void THNN_(HardTanh_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real min_val,
          real max_val);

TH_API void THNN_(L1Cost_updateOutput)(
          THNNState *state,
          THTensor *input,
          real *output);
TH_API void THNN_(L1Cost_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(LogSigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *buffer);
TH_API void THNN_(LogSigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *buffer);

TH_API void THNN_(LogSoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(LogSoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(LookupTable_accGradParameters)(
          THNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          real lr,
          bool shouldScaleGradByFreq, 
          THIndexTensor* count);

TH_API void THNN_(MarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          real *output,
          real margin,
          bool sizeAverage);
TH_API void THNN_(MarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          real margin,
          bool sizeAverage);

TH_API void THNN_(MSECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          real *output,
          bool sizeAverage);
TH_API void THNN_(MSECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(MultiLabelMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          real *output,
          bool sizeAverage);
TH_API void THNN_(MultiLabelMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(MultiMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          real* output,
          bool sizeAverage,
          int p);
TH_API void THNN_(MultiMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          int p);

TH_API void THNN_(PReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          long nOutputPlane);
TH_API void THNN_(PReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          long nOutputPlane);
TH_API void THNN_(PReLU_accGradParameters)(
          THNNState* state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradInput,
          THTensor *weight,
          THTensor *gradWeight,
          long nOutputPlane,
          real scale);

TH_API void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          real lower, 
          real upper,
          bool train,
          bool inplace,
          THGenerator *generator);
TH_API void THNN_(RReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          real lower,
          real upper,
          bool train,
          bool inplace);

TH_API void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(SoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(SoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);         
TH_API void THNN_(SpatialConvolution_updateOutput)(
          THNNState *state,
          THTensor* input,
          THTensor* output,
          THTensor* weight,
          THTensor* bias,
          THTensor* finput,
          int kW,   int kH,
          int dW,   int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolution_updateGradInput)(
          THNNState *state,
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
          THNNState *state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradWeight,
          THTensor* gradBias,
          THTensor* finput,
          real scale);

TH_API void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState *state,
          THTensor* input,
          THTensor* output,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int ceil_mode,
          THTensor* indices);
TH_API void THNN_(SpatialMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradInput,
          int dW, int dH,
          THTensor* indices);

TH_API void THNN_(Sqrt_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Sqrt_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(Square_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Square_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor* input,
          THTensor* output,
          real val,
          real threshold,
          int inPlace);

TH_API void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradInput,
          real threshold,
          int inPlace);
]]

local temp_str = {}

base_str = string.gsub(base_str, "TH_API void THNN_%(([%a%d_]+)%)", 'void THNN_TYPE%1') -- conversion used for generic/THNN.h

base_str = string.gsub(base_str, 'THIndexTensor','THLongTensor')

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

