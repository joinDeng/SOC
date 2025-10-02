#!/bin/bash

PYTHON_CMD="python3"

# 根据实际情况设置路径
DATA_DIR="/mnt/data/dyl/db"
HDF5="${DATA_DIR}/ncf_20220101-20230101.h5"
METRICS="${DATA_DIR}/space_object_metrics.json"

# 创建输出目录
mkdir -p ${DATA_DIR}/intermediate
mkdir -p ${DATA_DIR}/output

LIB_DIR="/mnt/data/dyl/project/SOC/src/lib"
pushd ${LIB_DIR}

# 步骤1: 分层抽样NORAD ID
echo "步骤1: 分层抽样NORAD ID"
${PYTHON_CMD} stratify_ids.py \
    --metrics ${METRICS} \
    --output ${DATA_DIR}/intermediate/selected_ids.json \
    --rare_output ${DATA_DIR}/intermediate/rare_ids.json

# 步骤2: 构建样本级索引
echo "步骤2: 构建样本级索引"
${PYTHON_CMD} build_sample_index.py \
    --h5_file ${HDF5} \
    --selected_ids ${DATA_DIR}/intermediate/selected_ids.json \
    --rare_ids ${DATA_DIR}/intermediate/rare_ids.json \
    --metrics ${METRICS} \
    --output ${DATA_DIR}/intermediate/sample_index.json

# 步骤3: 时间划分样本
echo "步骤3: 时间划分样本"
${PYTHON_CMD} split_samples.py \
    --h5_file ${HDF5} \
    --sample_index ${DATA_DIR}/intermediate/sample_index.json \
    --output_prefix ${DATA_DIR}/intermediate/split_samples

# 步骤4: 生成HDF5数据集
echo "步骤4: 生成训练集"
${PYTHON_CMD} write_splits.py \
    --h5_file ${HDF5} \
    --sample_index ${DATA_DIR}/intermediate/split_samples_train.json \
    --split_name train \
    --output ${DATA_DIR}/output/train.h5

echo "步骤4: 生成验证集"
${PYTHON_CMD} write_splits.py \
    --h5_file ${HDF5} \
    --sample_index ${DATA_DIR}/intermediate/split_samples_val.json \
    --split_name val \
    --output ${DATA_DIR}/output/val.h5

echo "步骤4: 生成测试集"
${PYTHON_CMD} write_splits.py \
    --h5_file ${HDF5} \
    --sample_index ${DATA_DIR}/intermediate/split_samples_test.json \
    --split_name test \
    --output ${DATA_DIR}/output/test.h5

echo "Pipeline完成!"
