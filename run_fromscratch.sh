#! /bin/bash

for name in scan_OMNI_0/*_run.json; do
    bname=${name%_run.json}
    echo $bname
    python few_shot_conv4_fromscratch.py $bname
done
