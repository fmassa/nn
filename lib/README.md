##THNN

###Compiling
From the `THNN` folder, run
```
cmake .
make
```

to check that it's working, run from the `lua` folder
```
th tests/test.lua
```

To include it in a th session, from the `lua` folder do
```lua
dofile 'init.lua' -- hack for the moment
```
