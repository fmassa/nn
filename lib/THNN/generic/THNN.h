#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THNN.h"
#else

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
          THLongTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);
TH_API void THNN_(ClassNLLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THLongTensor *target,
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
          THTensor *gradInput);

TH_API void THNN_(LogSigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *buffer);
TH_API void THNN_(LogSigmoid_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *buffer);

TH_API void THNN_(LogSoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(LogSoftMax_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(LookupTable_accGradParameters)(
          THNNState *state,
          THLongTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          real lr,
          bool shouldScaleGradByFreq, 
          THIntTensor* count);

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

TH_API void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor* input,
          THTensor* output,
          real val,
          real threshold,
          bool inPlace);

TH_API void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor* input,
          THTensor* gradOutput,
          THTensor* gradInput,
          real threshold,
          bool inPlace);

#endif
