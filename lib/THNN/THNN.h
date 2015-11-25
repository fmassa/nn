#ifndef THNN_H
#define THNN_H

#include <stdbool.h>

typedef void THNNState;

typedef struct THNN_FilterStruct
{
  long nOutputPlane;
  long nInputPlane;
  long kH;
  long kW;
} THNN_FilterStruct;

typedef struct THNN_ConvolutionStruct
{
  long dH;
  long dW;
  long padH;
  long padW;
} THNN_ConvolutionStruct;

//typedef THNN_FilterStruct* THNN_FilterDescriptor_t;
typedef THNN_FilterStruct THNN_FilterDescriptor_t;
//typedef THNN_ConvolutionStruct* THNN_ConvolutionDescriptor_t;
typedef THNN_ConvolutionStruct THNN_ConvolutionDescriptor_t;

void THNN_CreateFilterDescriptor(THNN_FilterDescriptor_t *filterDesc)
{
  filterDesc = THAlloc(sizeof(THNN_FilterStruct));
}

void THNN_SetFilterDescriptor(THNN_FilterDescriptor_t *filterDesc,
                              long nOutputPlane,
                              long nInputPlane,
                              long kH,
                              long kW
                              )
{
  filterDesc->nOutputPlane = nOutputPlane;
  filterDesc->nInputPlane = nInputPlane;
  filterDesc->kH = kH;
  filterDesc->kW = kW;
}

void THNN_DestroyFilterDescriptor(THNN_FilterDescriptor_t *filterDesc)
{
  THFree(filterDesc);
}

void THNN_CreateConvolutionDescriptor(THNN_ConvolutionDescriptor_t *convDesc)
{
  convDesc = THAlloc(sizeof(THNN_ConvolutionStruct));
}

void THNN_SetConvolutionDescriptor(THNN_ConvolutionDescriptor_t *convDesc,
                              long dH,
                              long dW,
                              long padH,
                              long padW
                              )
{
  convDesc->dH = dH;
  convDesc->dW = dW;
  convDesc->padH = padH;
  convDesc->padW = padW;
}

void THNN_DestroyConvolutionDescriptor(THNN_ConvolutionDescriptor_t *convDesc)
{
  THFree(convDesc);
}

#define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME)

#include "generic/THNN.h"
#include <THGenerateFloatTypes.h>

#endif
