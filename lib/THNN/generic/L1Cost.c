#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/L1Cost.c"
#else

void THNN_(L1Cost_updateOutput)(THNNState *state, THTensor *input, real *output)
{
  accreal sum = 0;

  TH_TENSOR_APPLY(real, input, 
    sum += fabs(*input_data);
  );

  *output = sum;
}

void THNN_(L1Cost_updateGradInput)(THNNState *state, THTensor *input, THTensor *gradInput)
{
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY2(real, gradInput, real, input,
    if (*input_data > 0)
      *gradInput_data = 1;
    else if (*input_data < 0)
      *gradInput_data = -1;
    else
      *gradInput_data = 0;
  );
}

#endif
