#!/bin/bash

# Usage:
#
# ./extract_med.sh
#

# if [ ! -d MED ]; then
#   git clone https://github.com/verypluming/MED.git
#   cd MED
#   git checkout 981440277c6b1c148c917cf813607b4bfdd0a892
#   cd ..
#   # echo "cd MED"
#   # echo "cp MED.tsv <parsing_comp directory>"
# fi

# TODO: change med, plain_dir, printf
med="./multinli_1.0/multinli_1.0_train.txt"

plain_dir="plain_multinli_train"

# Extract training and test data from MED dataset, removing the header line.
if [ ! -d ${plain_dir} ]; then
  mkdir -p ${plain_dir}
fi

echo "Extracting problems from the MNLI file."
cat $med | \
tr -d '\r' | \
awk -F'\t' -v tdir=${plain_dir} \
  '{pair_id=$1;
    id=NR;
    premise=toupper(substr($6,0,1))substr($6,2,length($6));
    conclusion=toupper(substr($7,0,1))substr($7,2,length($7));
    if($1 == "contradiction"){
      judgement="no";
    } else if ($1 == "entailment") {
      judgement="yes";
    } else if ($1 == "neutral") {
      judgement="unknown";
    }
    printf "%s\n%s\n", premise, conclusion > tdir"/mnli_train_"id"_gq.txt";
    printf "%s\n", judgement > tdir"/mnli_train_"id"_gq.answer";
   }'

