#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THNN.h"
#else




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
