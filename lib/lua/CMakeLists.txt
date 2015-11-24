FILE(GLOB luasrc *.lua)
INSTALL(FILES ${luasrc} 
  DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/nn/lib/lua)