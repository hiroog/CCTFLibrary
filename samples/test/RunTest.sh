#!/bin/sh

python3 SaveLoadTest_Keras.py --download
python3 ModelDefinition.py
flmake
if [ -e test_x64_Debug ]; then
./test_x64_Debug
fi
if [ -e test_arm64_Debug ]; then
./test_arm64_Debug
fi
