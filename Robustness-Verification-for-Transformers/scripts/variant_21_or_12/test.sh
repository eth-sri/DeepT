#!/usr/bin/env bash

nlayers=6

if [ $nlayers -eq 3 ]; then
    startFast=3
else
    startFast=4
fi

echo $startFast