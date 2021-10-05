#!/bin/zsh

# exec autopep8 for all .py file which is in the $1 directory.
# the {} will replace the found file .py
find ./$1 -name '*.py' -exec autopep8 --in-place '{}' \;
