@echo off

python SaveLoadTest_Keras.py --download
python ModelDefinition.py
flmake
test_x64_Debug.exe

