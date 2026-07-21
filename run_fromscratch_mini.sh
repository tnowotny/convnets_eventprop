#! /bin/bash

for name in scan_MINI_0/*_run.json; do
    bname=${name%_run.json}
    echo $bname
    python few_shot_conv4_fromscratch_mini_imagenet.py $bname
done
