#!/bin/bash
cd docs
rm ilustrado*.rst modules.rst
cd ../
sphinx-apidoc -o docs ilustrado setup.py ilustrado/tests/*
cd docs
make html
