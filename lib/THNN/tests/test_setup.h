#undef UPDATE_OUTPUT_FN
#undef UPDATE_GRAD_INPUT_FN
#undef ACC_GRAD_PARAMETERS_FN
#undef UPDATE_OUTPUT
#undef UPDATE_GRAD_INPUT
#undef ACC_GRAD_PARAMETERS
#undef ZERO_GRAD_PARAMETERS
#undef TEST
#undef TEST_GRAD_PARAMETERS
#undef CALC_NUMERICAL_JACOBIAN
#undef CALC_ANALYTICAL_JACOBIAN
#undef _STR
#undef STR


#define _STR(X) #X
#define STR(X) _STR(X)


#define UPDATE_OUTPUT_FN TH_CONCAT_4(THNN_,Real,MODULE_NAME,_updateOutput)
#define UPDATE_GRAD_INPUT_FN TH_CONCAT_4(THNN_,Real,MODULE_NAME,_updateGradInput)
#define ACC_GRAD_PARAMETERS_FN TH_CONCAT_4(THNN_,Real,MODULE_NAME,_accGradParameters)


#ifdef UPDATE_OUTPUT_ARGS
#  define UPDATE_OUTPUT(OUTPUT) UPDATE_OUTPUT_FN(NULL, input, OUTPUT, UPDATE_OUTPUT_ARGS)
#else
#  define UPDATE_OUTPUT(OUTPUT) UPDATE_OUTPUT_FN(NULL, input, OUTPUT)
#endif


#ifdef UPDATE_GRAD_INPUT_ARGS
#define UPDATE_GRAD_INPUT UPDATE_GRAD_INPUT_FN(NULL, input, dout, dinput, UPDATE_GRAD_INPUT_ARGS)
#else
#define UPDATE_GRAD_INPUT UPDATE_GRAD_INPUT_FN(NULL, input, dout, dinput)
#endif


#ifdef HAS_PARAMETERS
#  ifdef ACC_GRAD_PARAMETERS_ARGS
#    define ACC_GRAD_PARAMETERS ACC_GRAD_PARAMETERS_FN(NULL, input, dout, ACC_GRAD_PARAMETERS_ARGS)
#  else
#    define ACC_GRAD_PARAMETERS ACC_GRAD_PARAMETERS_FN(NULL, input, dout)
#  endif
#else
#  define ACC_GRAD_PARAMETERS {}
#endif


#ifdef HAS_PARAMETERS
#  define ZERO_GRAD_PARAMETERS                                                 \
{                                                                              \
  for (int j = 0; j < num_params; j++)                                         \
    THTensor_(fill)(gradParams[j], 0);                                         \
}
#else
#  define ZERO_GRAD_PARAMETERS {}
#endif


#define TEST                                                                   \
{                                                                              \
  printf("=============== " STR(MODULE_NAME) "\n");                            \
  long num_params = sizeof(params)/sizeof(THTensor *);                         \
  THTensor *dnumerical, *danalytical;                                          \
  THTensor *jacobian_numerical = THTensor_(new)();                             \
  THTensor *jacobian_analytical = THTensor_(new)();                            \
  THTensor *dinput = THTensor_(new)();                                         \
  /* numerical jacobian will be calculated w.r.t. dnumerical */                \
  dnumerical = input;                                                          \
  /* danalytical will be compared with numerical results */                    \
  danalytical = dinput;                                                        \
  printf("gradInput:\t");                                                      \
                                                                               \
  CALC_NUMERICAL_JACOBIAN;                                                     \
  CALC_ANALYTICAL_JACOBIAN;                                                    \
  COMPARE_JACOBIANS;                                                           \
                                                                               \
  TEST_GRAD_PARAMETERS;                                                        \
                                                                               \
  THTensor_(free)(jacobian_analytical);                                        \
  THTensor_(free)(jacobian_numerical);                                         \
  THTensor_(free)(dinput);                                                     \
}


#ifdef HAS_PARAMETERS
#define TEST_GRAD_PARAMETERS                                                   \
  {                                                                            \
    for (int i = 0; i < num_params; i++) {                                     \
      dnumerical = params[i];                                                  \
      danalytical = gradParams[i];                                             \
      printf("%s:\t", gradNames[i]);                                           \
      CALC_NUMERICAL_JACOBIAN;                                                 \
      CALC_ANALYTICAL_JACOBIAN;                                                \
      COMPARE_JACOBIANS;                                                       \
    }                                                                          \
  }
#else
#define TEST_GRAD_PARAMETERS {}
#endif


#define COMPARE_JACOBIANS                                                      \
{                                                                              \
  double result = maxDiff(jacobian_numerical, jacobian_analytical);            \
  if (result < precision)                                                      \
    printf("OK\n");                                                            \
  else                                                                         \
    printf("ERROR\n");                                                         \
}


#define CALC_NUMERICAL_JACOBIAN                                                \
{                                                                              \
  long dsize = THTensor_(nElement)(dnumerical);                                \
                                                                               \
  THTensor *d_vec = THTensor_(newWithStorage1d)(                               \
      THTensor_(storage)(dnumerical), 0, dsize, 1);                            \
                                                                               \
  THTensor *output = THTensor_(new)();                                         \
                                                                               \
  UPDATE_OUTPUT(output);                                                       \
                                                                               \
  long output_size = THTensor_(nElement)(output);                              \
  THTensor_(resize2d)(jacobian_numerical, dsize, output_size);                 \
  THTensor *out_plus = THTensor_(new)();                                       \
  THTensor *out_minus = THTensor_(new)();                                      \
  THTensor *jacobian_row = THTensor_(new)();                                   \
                                                                               \
  for(long i = 0; i < dsize; i++) {                                            \
    double orig = THTensor_(get1d)(d_vec, i);                                  \
    THTensor_(set1d)(d_vec, i, orig + perturbation);                           \
    UPDATE_OUTPUT(out_plus);                                                   \
    THTensor_(set1d)(d_vec, i, orig - perturbation);                           \
    UPDATE_OUTPUT(out_minus);                                                  \
    THTensor_(set1d)(d_vec, i, orig);                                          \
                                                                               \
    THTensor_(csub)(out_plus, out_plus, 1, out_minus);                         \
    THTensor_(div)(out_plus, out_plus, 2*perturbation);                        \
                                                                               \
    THTensor_(select)(jacobian_row, jacobian_numerical, 0, i);                 \
    THTensor_(copy)(jacobian_row, out_plus);                                   \
  }                                                                            \
                                                                               \
  THTensor_(free)(out_plus);                                                   \
  THTensor_(free)(out_minus);                                                  \
  THTensor_(free)(jacobian_row);                                               \
  THTensor_(free)(d_vec);                                                      \
  THTensor_(free)(output);                                                     \
}


#define CALC_ANALYTICAL_JACOBIAN                                               \
{                                                                              \
  THTensor *output = THTensor_(new)();                                         \
                                                                               \
  /* create a tensor for gradOutput */                                         \
  UPDATE_OUTPUT(output);                                                       \
  THTensor *dout = THTensor_(new)();                                           \
  THTensor_(resizeAs)(dout, output);                                           \
  THTensor_(fill)(dout, 0);                                                    \
                                                                               \
  /* dout_vec is a 1d view onto gradOutput */                                  \
  long output_size = THTensor_(nElement)(dout);                                \
  THTensor *dout_vec = THTensor_(newWithStorage1d)(                            \
      THTensor_(storage)(dout), 0, output_size, 1);                            \
                                                                               \
  /* reshape jacobian */                                                       \
  UPDATE_GRAD_INPUT;                                                           \
  long dsize = THTensor_(nElement)(danalytical);                               \
  THTensor_(resize2d)(jacobian_analytical, dsize, output_size);                \
  THTensor *jacobian_column = THTensor_(new)();                                \
                                                                               \
  for (long i = 0; i < output_size; i++) {                                     \
    THTensor_(set1d)(dout_vec, i, 1);                                          \
                                                                               \
    ZERO_GRAD_PARAMETERS;                                                      \
    UPDATE_GRAD_INPUT;                                                         \
    ACC_GRAD_PARAMETERS;                                                       \
                                                                               \
    THTensor_(select)(jacobian_column, jacobian_analytical, 1, i);             \
    THTensor_(copy)(jacobian_column, danalytical);                             \
    THTensor_(set1d)(dout_vec, i, 0);                                          \
  }                                                                            \
                                                                               \
  THTensor_(free)(output);                                                     \
  THTensor_(free)(dout);                                                       \
  THTensor_(free)(jacobian_column);                                            \
  THTensor_(free)(dout_vec);                                                   \
}
