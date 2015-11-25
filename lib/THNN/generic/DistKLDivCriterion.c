#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/DistKLDivCriterion.c"
#else

void THNN_(DistKLDivCriterion_updateOutput)(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, real *output)
{
  real sum;

  sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
                   sum += *target_data > 0 ? *target_data * (log(*target_data) - *input_data) : 0;)

  if(sizeAverage)
    sum /= THTensor_(nElement)(input);

  *output = sum;
}

void THNN_(DistKLDivCriterion_updateGradInput)(THNNState* state, THTensor *input, THTensor *target, bool sizeAverage, THTensor *gradInput)
{
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                   *gradInput_data = *target_data > 0 ? norm * (-*target_data) : 0;)
}

#endif
