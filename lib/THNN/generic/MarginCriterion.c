#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MarginCriterion.c"
#else

void THNN_(MarginCriterion_updateOutput)(THNNState *state, THTensor *input, THTensor *target, real *output, real margin, bool sizeAverage)
{ 
  real sum = 0;

  TH_TENSOR_APPLY2(real, input, real, target,
    real z = (margin - *input_data * *target_data);
    sum += z>0 ? z : 0;
  )

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  *output = sum;
}

void THNN_(MarginCriterion_updateGradInput)(THNNState *state, THTensor *input, THTensor *target, THTensor *gradInput, real margin, bool sizeAverage)
{
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    *gradInput_data = (*input_data * *target_data) < margin ? -norm * *target_data : 0;
  )
}

#endif
