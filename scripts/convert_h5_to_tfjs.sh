#!/bin/bash
model_path=$1
output_file=$2

tensorflowjs_converter --input_format keras $model_path $output_file