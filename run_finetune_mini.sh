#! /bin/bash

for name in scan_MINI_0/J0_?_checkpoints scan_MINI_0/J0_??_checkpoints scan_MINI_0/J0_???_checkpoints; do
    bname=${name%_checkpoints}
    echo $bname
    python few_shot_conv4_finetune_mini_imagenet.py $bname
done
