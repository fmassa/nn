#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THNN.h"
#else

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

#endif
