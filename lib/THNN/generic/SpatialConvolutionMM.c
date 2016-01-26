#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMM.c"
#else

#ifdef _WIN32
# include <windows.h>
#endif

/* note: due to write issues, this one cannot be parallelized as well as unfolded_copy */
static void THNN_(unfolded_acc)(THTensor *finput, THTensor *input,
                               int kW, int kH,
                               int dW, int dH,
                               int padW, int padH,
                               int nInputPlane,
                               int inputWidth, int inputHeight,
                               int outputWidth, int outputHeight)
{
#ifdef _WIN32
  LONG_PTR nip;
#else
  size_t nip;
#endif

  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(nip)
  for(nip = 0; nip < nInputPlane; nip++)
  {
    size_t kw, kh, y, x; 
    long long ix = 0, iy = 0;
    for(kh = 0; kh < kH; kh++)
    {
      for(kw = 0; kw < kW; kw++)
      {
        real *src = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
        real *dst = input_data + nip*(inputHeight*inputWidth);
        if (padW > 0 || padH > 0) {
          size_t lpad,rpad;
          for(y = 0; y < outputHeight; y++) {
            iy = (long long)(y*dH - padH + kh);
            if (iy < 0 || iy >= inputHeight) {
            } else {
              if (dW==1){
                 ix = (long long)(0 - padW + kw);
                 lpad = fmaxf(0,padW-kw);
                 rpad = fmaxf(0,padW-(kW-kw-1));
                 THVector_(add)(dst+(size_t)(iy*inputWidth+ix+lpad), src+(size_t)(y*outputWidth+lpad), 1, outputWidth - lpad - rpad); /* note: THVector_add could handle 1 value better */
              }
              else{
                for (x=0; x<outputWidth; x++){
                   ix = (long long)(x*dW - padW + kw);
                   if (ix < 0 || ix >= inputWidth){
                   }else
                     THVector_(add)(dst+(size_t)(iy*inputWidth+ix), src+(size_t)(y*outputWidth+x), 1, 1);
                }
              }
            }
          }
        } else {
          for(y = 0; y < outputHeight; y++) {
            iy = (long long)(y*dH + kh);
            ix = (long long)(0 + kw);
            if (dW == 1 )
               THVector_(add)(dst+(size_t)(iy*inputWidth+ix), src+(size_t)(y*outputWidth), 1, outputWidth); /* note: THVector_add could handle 1 value better */
            else{
              for(x = 0; x < outputWidth; x++)
                THVector_(add)(dst+(size_t)(iy*inputWidth+ix+x*dW), src+(size_t)(y*outputWidth+x), 1, 1);
            }
          }
        }
      }
    }
  }
}

static void THNN_(unfolded_copy)(THTensor *finput, THTensor *input,
                               int kW, int kH,
                               int dW, int dH,
                               int padW, int padH,
                               int nInputPlane,
                               int inputWidth, int inputHeight,
                               int outputWidth, int outputHeight)
{
  long k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane*kH*kW; k++) {
    size_t nip = k / (kH*kW);
    size_t rest = k % (kH*kW);
    size_t kh = rest / kW;
    size_t kw = rest % kW;
    size_t x,y;
    long long ix,iy;
    real *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
    real *src = input_data + nip*(inputHeight*inputWidth);
    if (padW > 0 || padH > 0) {
      size_t lpad,rpad;
      for(y = 0; y < outputHeight; y++) {
        iy = (long long)(y*dH - padH + kh);
        if (iy < 0 || iy >= inputHeight) {
          memset(dst+y*outputWidth, 0, sizeof(real)*outputWidth);
        } else {
          if (dW==1){
             ix = (long long)(0 - padW + kw);
             lpad = fmaxf(0,padW-kw);
             rpad = fmaxf(0,padW-(kW-kw-1));
             if (outputWidth-rpad-lpad <= 0) {
                memset(dst+(size_t)(y*outputWidth), 0, sizeof(real)*outputWidth);
             } else {
                if (lpad > 0) memset(dst+y*outputWidth, 0, sizeof(real)*lpad);
                memcpy(dst+(size_t)(y*outputWidth+lpad), src+(size_t)(iy*inputWidth+ix+lpad), sizeof(real)*(outputWidth-rpad-lpad));
                if (rpad > 0) memset(dst+y*outputWidth + outputWidth - rpad, 0, sizeof(real)*rpad);
             }
          }
          else{
            for (x=0; x<outputWidth; x++){
               ix = (long long)(x*dW - padW + kw);
               if (ix < 0 || ix >= inputWidth)
                 memset(dst+(size_t)(y*outputWidth+x), 0, sizeof(real)*1);
               else
                 memcpy(dst+(size_t)(y*outputWidth+x), src+(size_t)(iy*inputWidth+ix), sizeof(real)*(1));
            }
          }
        }
      }
    } else {
      for(y = 0; y < outputHeight; y++) {
        iy = (long long)(y*dH + kh);
        ix = (long long)(0 + kw);
        if (dW == 1)
           memcpy(dst+(size_t)(y*outputWidth), src+(size_t)(iy*inputWidth+ix), sizeof(real)*outputWidth);
        else{
          for (x=0; x<outputWidth; x++)
             memcpy(dst+(size_t)(y*outputWidth+x), src+(size_t)(iy*inputWidth+ix+x*dW), sizeof(real)*(1));
         }
      }
    }
  }
}

static void THNN_(SpatialConvolutionMM_updateOutput_frame)(THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, THTensor *finput,
                                                         int kW, int kH, int dW, int dH, int padW, int padH,
                                                         long nInputPlane, long inputWidth, long inputHeight,
                                                         long nOutputPlane, long outputWidth, long outputHeight)
{
  long i;
  THTensor *output2d;

  THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(THTensor_(data)(output)+output->stride[0]*i, THTensor_(get1d)(bias, i), outputHeight*outputWidth);

  // M,N,K are dims of matrix A and B
  // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
  long m = nOutputPlane;
  long n = finput->size[1];
  long k = nInputPlane*kH*kW;

  // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
  THBlas_(gemm)(
      'n', 'n',
      n, m, k,
      1,
      THTensor_(data)(finput), n,
      THTensor_(data)(weight), k,
      1,//0,//1,
      THTensor_(data)(output), n
  );

  /*
  real *output_data = THTensor_(data)(output);
  for(i = 0; i < nOutputPlane; i++)
  {
    real *output_data0 = output_data + output->stride[0]*i;
    real bias0 = THTensor_(get1d)(bias,i);
    for(long j = 0; j < outputHeight*outputWidth; j++) {
        *(output_data0++) += bias0;
    }
 
  }
  */
}

void THNN_(SpatialConvolution_forwardFilter)(THNNState *state, THTensor *input, THTensor *output, THTensor *weight, THTensor* columns, int dW, int dH, int padW, int padH)
{
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 1, "3D or 4D (batch mode) tensor expected");
  THArgCheck(weight->nDimension == 4, 4, "weight tensor must be 4D (nOutputPlane,nInputPlane,kH,kW)");

  long nInputPlane = weight->size[1];
  long nOutputPlane = weight->size[0];
  int kH = weight->size[2];
  int kW = weight->size[3];

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
  }
  THArgCheck(input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");

  long inputWidth;
  long inputHeight;
  long outputWidth;
  long outputHeight;

  inputWidth   = input->size[3];
  inputHeight  = input->size[2];
  outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  long batchSize = input->size[0];
  long t;

  THTensor_(resize3d)(columns, batchSize, kW*kH*nInputPlane, outputHeight*outputWidth);
  THTensor_(resize4d)(output, batchSize, nOutputPlane, outputHeight, outputWidth);

#pragma omp parallel for private(t) if(batchSize > 1)
  for(t = 0; t < batchSize; t++)
  {
    THTensor *input_t = THTensor_(newSelect)(input, 0, t);
    THTensor *output_t = THTensor_(newSelect)(output, 0, t);
    THTensor *columns_t = THTensor_(newSelect)(columns, 0, t);

    THNN_(unfolded_copy)(columns_t, input_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = outputHeight*outputWidth;//columns_t->size[1];
    long k = nInputPlane*kH*kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
        'n', 'n',
        n, m, k,
        1,
        THTensor_(data)(columns_t), n,
        THTensor_(data)(weight), k,
        0,
        THTensor_(data)(output_t), n
    );

    THTensor_(free)(input_t);
    THTensor_(free)(output_t);
    THTensor_(free)(columns_t);
  }

  // Resize output
  if (batch == 0) {
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
  }

}


void THNN_(SpatialConvolution_addBiasForward)(THNNState *state, THTensor *output, THTensor *bias, THTensor *ones)
{
  int nOutputPlane = bias->size[0];
  int batch = 1;
  if (output->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(output, 1, output->size[0], output->size[1], output->size[2]);
  }
  THArgCheck(output->size[1] == nOutputPlane, 2, "input channels and nInputPlane dont match");

  int batchSize = output->size[0];
  long outputHeight = output->size[2];
  long outputWidth = output->size[3];
  long t, i, j;
//#pragma omp parallel for private(t) if(batchSize > 1)
  for(t = 0; t < batchSize; t++)
  {
    THTensor *output_t = THTensor_(newSelect)(output, 0, t);
    real *output_data = THTensor_(data)(output_t);
    for(i = 0; i < nOutputPlane; i++)
    {
      real *output_data0 = output_data + output->stride[1]*i;
      real bias0 = THTensor_(get1d)(bias,i);
      for(j = 0; j < outputHeight*outputWidth; j++) {
        *(output_data0++) += bias0;
      }
    }
    THTensor_(free)(output_t);
  }

  // Resize output
  if (batch == 0) {
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
  }

}

void THNN_(SpatialConvolutionMM_updateOutput)(THNNState *state, THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, THTensor* finput, int kW, int kH, int dW, int dH, int padW, int padH)
{
  int dimf = 0;
  int dimw = 2;
  int dimh = 1;

  long nInputPlane;
  long inputWidth;
  long inputHeight;
  long nOutputPlane;
  long outputWidth;
  long outputHeight;

  THArgCheck( input->nDimension == 3 || input->nDimension == 4, 1, "3D or 4D (batch mode) tensor expected");

  if (input->nDimension == 4) {
    dimf++;
    dimw++;
    dimh++;
  }

  nInputPlane = input->size[dimf];
  inputWidth   = input->size[dimw];
  inputHeight  = input->size[dimh];
  nOutputPlane = weight->size[0];
  outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  //if (nInputPlane*kW*kH != weight->size[1])
    //THError("Wrong number of input channels! Input has %d channels, expected %d",nInputPlane,weight->size[1]/(kW*kH));

  if(input->nDimension == 3)
  {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    THNN_(SpatialConvolutionMM_updateOutput_frame)(input, output, weight, bias, finput,
                                                 kW, kH, dW, dH, padW, padH,
                                                 nInputPlane, inputWidth, inputHeight,
                                                 nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(SpatialConvolutionMM_updateOutput_frame)(input_t, output_t, weight, bias, finput_t,
                                                   kW, kH, dW, dH, padW, padH,
                                                   nInputPlane, inputWidth, inputHeight,
                                                   nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }
}

void THNN_(SpatialConvolution_backwardData)(THNNState *state, THTensor *input, THTensor *gradInput, THTensor *gradOutput, THTensor *weight, THTensor *gradColumns, int dW, int dH, int padW, int padH)
{

  long nInputPlane = weight->size[1];
  long nOutputPlane = weight->size[0];
  int kH = weight->size[2];
  int kW = weight->size[3];

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  THArgCheck(nOutputPlane == gradOutput->size[1], 1, "Number of gradOutput features is not equal to nOutputPlane" );

  THTensor_(resizeAs)(gradInput, input);
  //THTensor_(resizeAs)(gradColumns, columns);

  THTensor_(zero)(gradInput);


  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = gradOutput->size[3];
  long outputHeight = gradOutput->size[2];

  long batchSize = input->size[0];
  long t;

#pragma omp parallel for private(t) if(batchSize > 1)
  for(t = 0; t < batchSize; t++)
  {
    THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
    THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
    THTensor *gradColumns_t = THTensor_(newSelect)(gradColumns, 0, t);


    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = nInputPlane*kW*kH;
    long n = gradColumns->size[1];
    long k = nOutputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
        'n', 't',
        n, m, k,
        1,
        THTensor_(data)(gradOutput_t), n,
        THTensor_(data)(weight), m,
        0,
        THTensor_(data)(gradColumns), n
    );


    THNN_(unfolded_acc)(gradColumns, gradInput_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

    THTensor_(free)(gradInput_t);
    THTensor_(free)(gradOutput_t);
    THTensor_(free)(gradColumns_t);
  }

  // Resize output
  if (batch == 0) {
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize3d)(gradInput, nInputPlane, inputHeight, inputWidth);
  }
}



static void THNN_(SpatialConvolutionMM_updateGradInput_frame)(THTensor *gradInput, THTensor *gradOutput, THTensor *weight, THTensor *fgradInput,
                                                            int kW, int kH, int dW, int dH, int padW, int padH)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);
  THTensor_(addmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
  THTensor_(free)(gradOutput2d);

  THTensor_(zero)(gradInput);

  THNN_(unfolded_acc)(fgradInput, gradInput, kW, kH, dW, dH, padW, padH, gradInput->size[0], gradInput->size[2], gradInput->size[1], gradOutput->size[2], gradOutput->size[1]);
}

void THNN_(SpatialConvolutionMM_updateGradInput)(THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput, THTensor *weight, THTensor *bias, THTensor *finput, THTensor *fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)
{
  long nOutputPlane = weight->size[0];

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  THTensor_(transpose)(weight, weight, 0, 1);

  if(input->nDimension == 3)
  {
    THNN_(SpatialConvolutionMM_updateGradInput_frame)(gradInput, gradOutput, weight, fgradInput, kW, kH, dW, dH, padW, padH);
  }
  else
  {
    long T = input->size[0];
    long t;

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

      THNN_(SpatialConvolutionMM_updateGradInput_frame)(gradInput_t, gradOutput_t, weight, fgradInput_t, kW, kH, dW, dH, padW, padH);

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

  THTensor_(transpose)(weight, weight, 0, 1);
}

static void THNN_(SpatialConvolutionMM_accGradParameters_frame)(THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, THTensor *finput,
                                                              real scale)
{
  long i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);

  THTensor_(transpose)(finput, finput, 0, 1);
  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, finput);
  THTensor_(transpose)(finput, finput, 0, 1);

  for(i = 0; i < gradBias->size[0]; i++)
  {
    long k;
    real sum = 0;
    real *data = gradOutput2d->storage->data + gradOutput2d->storageOffset + i*gradOutput2d->stride[0];
    for(k = 0; k < gradOutput2d->size[1]; k++)
      sum += data[k];
    (gradBias->storage->data + gradBias->storageOffset)[i] += scale*sum;
  }

  THTensor_(free)(gradOutput2d);
}

void THNN_(SpatialConvolution_backwardFilter)(THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradWeight, THTensor *columns, int dW, int dH, int padW, int padH, real scale)
{
  long nInputPlane = gradWeight->size[1];
  long nOutputPlane = gradWeight->size[0];
  int kH = gradWeight->size[2];
  int kW = gradWeight->size[3];

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  long inputHeight = input->size[2];
  long inputWidth = input->size[3];
  long outputHeight = gradOutput->size[2];
  long outputWidth = gradOutput->size[3];
  long batchSize = input->size[0];
  long t;

  for(t = 0; t < batchSize; t++)
  {
    THTensor *input_t = THTensor_(newSelect)(input, 0, t);
    THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
    THTensor *columns_t = THTensor_(newSelect)(columns, 0, t);

    THNN_(unfolded_copy)(columns_t, input_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = nInputPlane*kW*kH;
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
        't', 'n',
        n, m, k,
        scale,
        THTensor_(data)(columns), k,
        THTensor_(data)(gradOutput_t), k,
        1,
        THTensor_(data)(gradWeight), n
    );

    THTensor_(free)(input_t);
    THTensor_(free)(gradOutput_t);
    THTensor_(free)(columns_t);
  }
}

void THNN_(SpatialConvolution_backwardBias)(THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradBias, THTensor *ones, real scale)
{
  long nOutputPlane = gradBias->size[0];

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  long outputHeight = gradOutput->size[2];
  long outputWidth = gradOutput->size[3];

  long batchSize = input->size[0];
  long t;

  for(t = 0; t < batchSize; t++)
  {
    THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    THBlas_(gemv)(
        't',
        k_, m_,
        scale,
        THTensor_(data)(gradOutput_t), k_,
        THTensor_(data)(ones), 1,
        1,
        THTensor_(data)(gradBias), 1
    );

    THTensor_(free)(gradOutput_t);
  }
}

void THNN_(SpatialConvolutionMM_accGradParameters)(THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, THTensor *finput, real scale)
{
  long nOutputPlane = gradWeight->size[0];
  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  if(input->nDimension == 3)
  {
    THNN_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, scale);
  }
  else
  {
    long T = input->size[0];
    long t;

    for(t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, scale);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }
}

#endif
