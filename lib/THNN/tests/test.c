#include "THNN.h"

#define real double
#define Real Double
#define accreal double

double maxDiff(THTensor *t1, THTensor *t2)
{
  THTensor_(csub)(t1, t1, 1, t2);
  THTensor_(abs)(t1, t1);
  return THTensor_(maxall)(t1);
}


int main()
{
  double precision = 1e-5;
  double perturbation = 1e-6;
  THGenerator *prng = THGenerator_new();
  THTensor params[0];
  long num_params = 0;

  {
    THTensor *input = THTensor_(newWithSize2d)(3,3);
    THTensor_(uniform)(input, prng, -10, 10);
    #define MODULE_NAME Abs
    #include "test_setup.h"
    TEST;
    #include "test_teardown.h"
    THTensor_(free)(input);
  }

  {
    THTensor *input = THTensor_(newWithSize4d)(1, 1, 5, 5);
    THTensor_(uniform)(input, prng, 0, 1);
    THTensor *weight = THTensor_(newWithSize2d)(1, 25);
    THTensor_(uniform)(weight, prng, -0.5, 0.5);
    THTensor *bias = THTensor_(newWithSize1d)(1);
    THTensor_(uniform)(bias, prng, -0.5, 0.5);
    THTensor *finput = THTensor_(new)();
    THTensor *fgradinput = THTensor_(new)();
    THTensor *gradWeight = THTensor_(newWithSize2d)(1, 25);
    THTensor *gradBias = THTensor_(newWithSize1d)(1);
    THTensor_(uniform)(gradWeight, prng, -0.5, 0.5);
    THTensor_(uniform)(gradBias, prng, -0.5, 0.5);

    THTensor *params[] = {weight, bias};
    THTensor *gradParams[] = {gradWeight, gradBias};
    char *gradNames[] = {"gradWeight", "gradBias"};

    #define MODULE_NAME SpatialConvolution
    #define HAS_PARAMETERS
    #define UPDATE_OUTPUT_ARGS weight, bias, finput, 5, 5, 1, 1, 0, 0
    #define UPDATE_GRAD_INPUT_ARGS weight, bias, finput, fgradinput, 5, 5, 1, 1, 0, 0
    #define ACC_GRAD_PARAMETERS_ARGS gradWeight, gradBias, finput, 1
    #include "test_setup.h"
    TEST;
    #include "test_teardown.h"


    THTensor_(free)(input);
    THTensor_(free)(weight);
    THTensor_(free)(bias);
    THTensor_(free)(finput);
    THTensor_(free)(fgradinput);
    THTensor_(free)(gradWeight);
    THTensor_(free)(gradBias);
  }

  THGenerator_free(prng);

  return 0;
}

