#!/bin/bash
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [NEED_PREPROCESS] [DEVICE_ID]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)

dataset_path=$(get_real_path $2)

if [ "$3" == "y" ] || [ "$3" == "n" ];then
    need_preprocess=$3
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "mindir name: "$model
echo "dataset path: "$dataset_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --datapath=$dataset_path --result_path=./preprocess_Result/
}

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    rm -rf ./result_Files_all infer.log
    mkdir result_Files_all
    mkdir time_Result

    for i in {0..49}
    do
      mkdir result_Files
      ../ascend310_infer/out/main --mindir_path=$model --input0_path=./preprocess_Result/data_$i/00_users --input1_path=./preprocess_Result/data_$i/01_items --input2_path=./preprocess_Result/data_$i/02_neg_items --input3_path=./preprocess_Result/data_$i/03_u_test_neighs --input4_path=./preprocess_Result/data_$i/04_u_test_gnew_neighs --input5_path=./preprocess_Result/data_$i/05_i_test_neighs --input6_path=./preprocess_Result/data_$i/06_i_test_gnew_neighs --device_id=$device_id &>> infer.log
      mv result_Files result_Files_all/result_Files_$i
    done
}

function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files_all --datapath=$dataset_path &> acc.log
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess dataset failed"
        exit 1
    fi
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi