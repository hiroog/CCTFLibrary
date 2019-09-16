#!/bin/sh

python3 SaveLoadTest_Keras.py --download
python3 ModelDefinition.py
flmake
./test_x64_Debug
