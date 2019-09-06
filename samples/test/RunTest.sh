#!/bin/sh

python3 CreateModel.py
flmake
./test_x64_Debug
