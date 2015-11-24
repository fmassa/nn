#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/AbsCriterion.c"
#else

int THNN_(AbsCriterion_updateOutput)(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, double *output)
{
  real sum;

  sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
                   sum += fabs(*input_data - *target_data);)

  if(sizeAverage)
    sum /= THTensor_(nElement)(input);

  *output = (double)sum;
  
  return 0;
}

int THNN_(AbsCriterion_updateGradInput)(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, THTensor *gradInput)
{
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                   *gradInput_data = ( (*input_data - *target_data) >= 0 ? norm : -norm);)

  return 0;
}

#endif
