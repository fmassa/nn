#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricDeconvolution.c"
#else

void THNN_(VolumetricDeconvolution_updateOutput)(
  THNNState *state, THTensor *input, THTensor *output, THTensor *weight, THTensor *bias,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int pT, int pH, int pW,
  int nInputPlane, int nOutputPlane
)
{
  int inputDepth   = input->size[2];
  int inputHeight  = input->size[3];
  int inputWidth   = input->size[4];

  int outputDepth  = (inputDepth - 1) * dT - 2 * pT + kT;
  int outputHeight = (inputHeight - 1) * dH - 2 * pH + kH;
  int outputWidth  = (inputWidth - 1) * dW - 2 * pW + kW;

  THArgCheck(input->nDimension == 5, 2, "5D (batch mode) tensor is expected");
  THArgCheck(kH == kW && pH == pW, 2, "kH == kW && pH == pW is expected");

  // Batch size
  long batchSize = input->size[0];

  // Resize output
  THTensor_(resize5d)(output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  const real* weight_ptr = THTensor_(data)(weight);
  const real* bias_ptr = THTensor_(data)(bias);

  int n;
  for (n = 0; n < batchSize; ++n)
  {
    THTensor_(select)(input_n, input, 0, n);
    THTensor_(select)(output_n, output, 0, n);

    THTensor *outn = THTensor_(new)();
    // add bias first
    int i;
    for (i = 0; i < bias->size[0]; i++)
    {
      THTensor_(select)(outn,output_n,0,i);
      THTensor_(fill)(outn, THTensor_(get1d)(bias, i));
    }
    THTensor_(free)(outn);

    int t, h, w, kc_, kt_, kh_, kw_, c;

    const real* input_ptr = THTensor_(data)(input_n);
    real* output_ptr = THTensor_(data)(output_n);
    for (t = 0; t < inputDepth; t++)
      for (h = 0; h < inputHeight; h++)
        for (w = 0; w < inputWidth; w++)
          for (kc_ = 0; kc_ < nOutputPlane; kc_++)
            for (kt_ = 0; kt_ < kT; kt_++)
              for (kh_ = 0; kh_ < kH; kh_++)
                for (kw_ = 0; kw_ < kW; kw_++)
                {
                  int pt = t * dT - pT + kt_;
                  int ph = h * dH - pH + kh_;
                  int pw = w * dW - pW + kw_;
                  if (pt >=0 && ph >=0 && pw >= 0 && pt < outputDepth && ph < outputHeight && pw < outputWidth)
                  {
                    real val = 0;
                    for (c = 0; c < nInputPlane; c++)
                    {
                      val += input_ptr[((c * inputDepth + t) * inputHeight + h) * inputWidth + w]
                        * weight_ptr[(((kc_ * nInputPlane + c) * kT + kt_) * kH + kh_) * kW + kw_];
                    }
                    output_ptr[((kc_ * outputDepth + pt) * outputHeight + ph) * outputWidth + pw]
                      += val;
                  }
                }
  }
  
  THTensor_(free)(input_n);
  THTensor_(free)(output_n);
}

void THNN_(VolumetricDeconvolution_updateGradInput)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput, THTensor *weight, THTensor *bias,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int pT, int pH, int pW,
  int nInputPlane, int nOutputPlane
)
{
  int inputDepth   = input->size[2];
  int inputHeight  = input->size[3];
  int inputWidth   = input->size[4];

  int outputDepth  = (inputDepth - 1) * dT - 2 * pT + kT;
  int outputHeight = (inputHeight - 1) * dH - 2 * pH + kH;
  int outputWidth  = (inputWidth - 1) * dW - 2 * pW + kW;

  THArgCheck(input->nDimension == 5, 2, "5D (batch mode) tensor is expected");
  THArgCheck(kH == kW && pH == pW, 2, "kH == kW && pH == pW is expected");

  // Batch size
  long batchSize = input->size[0];

  // Resize output
  THTensor_(resize5d)(gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Helpers
  THTensor *gradInput_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  const real* weight_ptr = THTensor_(data)(weight);

  // For each n in batch, do:
  int n;
  for (n = 0; n < batchSize; n++)
  {
    THTensor_(select)(gradInput_n, gradInput, 0, n);
    THTensor_(select)(gradOutput_n, gradOutput, 0, n);
    THTensor_(fill)(gradInput_n, 0);

    int t, h, w, kc_, kt_, kh_, kw_, c;

    real* gradInput_ptr = THTensor_(data)(gradInput_n);
    const real* gradOutput_ptr = THTensor_(data)(gradOutput_n);
    for (t = 0; t < inputDepth; t++)
      for (h = 0; h < inputHeight; h++)
        for (w = 0; w < inputWidth; w++)
          for (kc_ = 0; kc_ < nOutputPlane; kc_++)
            for (kt_ = 0; kt_ < kT; kt_++)
              for (kh_ = 0; kh_ < kH; kh_++)
                for (kw_ = 0; kw_ < kW; kw_++)
                {
                  int pt = t * dT - pT + kt_;
                  int ph = h * dH - pH + kh_;
                  int pw = w * dW - pW + kw_;
                  if (pt >=0 && ph >=0 && pw >= 0 && pt < outputDepth && ph < outputHeight && pw < outputWidth)
                  {
                    for (c = 0; c < nInputPlane; c++)
                    {
                      gradInput_ptr[((c * inputDepth + t) * inputHeight + h) * inputWidth + w] +=
                        gradOutput_ptr[((kc_ * outputDepth + pt) * outputHeight + ph) * outputWidth + pw]
                        * weight_ptr[(((kc_ * nInputPlane + c) * kT + kt_) * kH + kh_) * kW + kw_];
                    }
                  }
                }
  }

  // Free
  THTensor_(free)(gradInput_n);
  THTensor_(free)(gradOutput_n);
}

void THNN_(VolumetricDeconvolution_accGradParameters)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, 
  int kT, int kH, int kW, 
  int dT, int dH, int dW, 
  int pT, int pH, int pW, 
  int nInputPlane, int nOutputPlane
)
{
  THArgCheck(input->nDimension == 5, 2, "5D (batch mode) tensor is expected");
  THArgCheck(kH == kW && pH == pW, 2, "kH == kW && pH == pW is expected");

  THTensor_(resize1d)(gradBias, nOutputPlane);
  THTensor_(resize5d)(gradWeight, nOutputPlane, nInputPlane, kT, kH, kW);

  int inputDepth   = input->size[2];
  int inputHeight  = input->size[3];
  int inputWidth   = input->size[4];

  int outputDepth  = (inputDepth - 1) * dT - 2 * pT + kT;
  int outputHeight = (inputHeight - 1) * dH - 2 * pH + kH;
  int outputWidth  = (inputWidth - 1) * dW - 2 * pW + kW;

  // Batch size
  long batchSize = input->size[0];

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  // reset gradBias = 0
  THTensor_(fill)(gradBias, 0);
  // reset gradWeight = 0
  THTensor_(fill)(gradWeight, 0);

  real* gradWeight_ptr = THTensor_(data)(gradWeight);
  real* gradBias_ptr = THTensor_(data)(gradBias);

  // For each n in batch, do:
  int n;
  for (n = 0; n < batchSize; n++)
  {
    THTensor_(select)(input_n, input, 0, n);
    THTensor_(select)(gradOutput_n, gradOutput, 0, n);

    THTensor *goutn = THTensor_(new)();

    // accumulate bias gradient first
    int i;
    for (i = 0; i < gradBias->size[0]; i++)
    {
      THTensor_(select)(goutn, gradOutput_n, 0, i);
      gradBias_ptr[i] += THTensor_(sumall)(goutn);
    }
    THTensor_(free)(goutn);

    int t, h, w, kc_, kt_, kh_, kw_, c;

    const real* input_ptr = THTensor_(data)(input_n);
    const real* gradOutput_ptr = THTensor_(data)(gradOutput_n);
    for (t = 0; t < inputDepth; t++)
      for (h = 0; h < inputHeight; h++)
        for (w = 0; w < inputWidth; w++)
          for (kc_ = 0; kc_ < nOutputPlane; kc_++)
            for (kt_ = 0; kt_ < kT; kt_++)
              for (kh_ = 0; kh_ < kH; kh_++)
                for (kw_ = 0; kw_ < kW; kw_++)
                {
                  int pt = t * dT - pT + kt_;
                  int ph = h * dH - pH + kh_;
                  int pw = w * dW - pW + kw_;
                  if (pt >=0 && ph >=0 && pw >= 0 && pt < outputDepth && ph < outputHeight && pw < outputWidth)
                  {
                    for (c = 0; c < nInputPlane; c++)
                    {
                      gradWeight_ptr[(((kc_ * nInputPlane + c) * kT + kt_) * kH + kh_) * kW + kw_] +=
                        input_ptr[((c * inputDepth + t) * inputHeight + h) * inputWidth + w] *
                        gradOutput_ptr[((kc_ * outputDepth + pt) * outputHeight + ph) * outputWidth + pw];
                    }
                  }
                }
  }

  // Free
  THTensor_(free)(input_n);
  THTensor_(free)(gradOutput_n);
}

#endif
