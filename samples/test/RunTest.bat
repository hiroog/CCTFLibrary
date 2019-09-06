@echo off

python CreateModel.py
flmake
test_x64_Debug.exe

