#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Square.c"
#else

static int nn_(Square_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THNNState *state;
  THNN_(Square_updateOutput)(state, input, output)

  return 1;
}

static int nn_(Square_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THNNState *state;
  THNN_(Square_updateGradInput)(state, input, gradOutput, gradInput)

  return 1;
}

static const struct luaL_Reg nn_(Square__) [] = {
  {"Square_updateOutput", nn_(Square_updateOutput)},
  {"Square_updateGradInput", nn_(Square_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Square_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Square__), "nn");
  lua_pop(L,1);
}

#endif
