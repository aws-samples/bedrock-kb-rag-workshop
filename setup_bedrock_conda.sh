#!/bin/bash

# constants
BEDROCK_CONDA_ENV=bedrock_py39
PY_VER=3.9

# create conda env
conda remove -n $BEDROCK_CONDA_ENV --all -y
conda create --name $BEDROCK_CONDA_ENV -y python=$PY_VER ipykernel
source activate $BEDROCK_CONDA_ENV

# all set to pip install the bedrock packages, these are awscli, boto3 and botocore
pip install --no-build-isolation --force-reinstall boto3>=1.28.57 awscli>=1.29.57 botocore>=1.31.57
pip install langchain==0.0.304
pip install opensearch-py==2.3.1

echo "all done"
