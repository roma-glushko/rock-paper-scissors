#!/bin/bash
output_dataset_name=$1
first_dataset_name=$2
second_dataset_name=$3

# merge two datasets to create a third one
# Command Sample: sh ./scripts/merge_datasets.sh rps-webcam-val-dataset rps/rps webcam_val

mkdir -p ./data/${output_dataset_name}/

cp -R ./data/${first_dataset_name}/* ./data/${output_dataset_name}/
cp -R ./data/${second_dataset_name}/* ./data/${output_dataset_name}/