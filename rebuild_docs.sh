#!/bin/bash
cd docs
rm ilustrado*.rst modules.rst
cd ../
sphinx-apidoc -o docs ilustrado setup.py
cd docs
make html
