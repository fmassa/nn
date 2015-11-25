#include <stdbool.h>
#include "TH.h"
#include "THNN.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)
#define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME)

#include "generic/THNN.h"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionMM.c"
#include "THGenerateFloatTypes.h"

#include "generic/Abs.c"
#include "THGenerateFloatTypes.h"

#include "generic/AbsCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/ClassNLLCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/DistKLDivCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/HardShrink.c"
#include "THGenerateFloatTypes.h"
