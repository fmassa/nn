#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THNN.h"
#else

TH_API void THNN_(Abs_updateOutput)(THNNState* state, THTensor *input, THTensor *output);
TH_API void THNN_(Abs_updateGradInput)(THNNState* state, THTensor *input, THTensor *gradOutput, THTensor *gradInput);

TH_API void THNN_(AbsCriterion_updateOutput)(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, real *output);
TH_API void THNN_(AbsCriterion_updateGradInput)(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, THTensor *gradInput);

TH_API void THNN_(ClassNLLCriterion_updateOutput)(THNNState* state, THTensor *input, THLongTensor *target, bool sizeAverage, THTensor *weights, THTensor *total_weight, THTensor *output);
TH_API void THNN_(ClassNLLCriterion_updateGradInput)(THNNState* state, THTensor *input, THLongTensor *target, bool sizeAverage, THTensor *weights, THTensor *total_weight, THTensor *gradInput);

TH_API void THNN_(DistKLDivCriterion_updateOutput)(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, real *output);
TH_API void THNN_(DistKLDivCriterion_updateGradInput)(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, THTensor *gradInput);

TH_API void THNN_(HardShrink_updateOutput)(THNNState* state, THTensor *input, real lambda, THTensor *output);
TH_API void THNN_(HardShrink_updateGradInput)(THNNState* state, THTensor *input, real lambda, THTensor *gradOutput, THTensor *gradInput);

TH_API void THNN_(SpatialConvolution_updateOutput)(THNNState* state,
                                           THTensor* input,
                                           THTensor* weight,
                                           THTensor* bias,
                                           THTensor* finput,
                                           THTensor* output,
                                           int kW,   int kH,
                                           int dW,   int dH,
                                           int padW, int padH);
TH_API void THNN_(SpatialConvolution_updateGradInput)(THNNState* state,
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
TH_API void THNN_(SpatialConvolution_accGradParameters)(THNNState* state,
                                           THTensor* input,
                                           THTensor* gradWeight,
                                           THTensor* gradBias,
                                           THTensor* gradOutput,
                                           THTensor* finput,
                                           real scale);

#endif
