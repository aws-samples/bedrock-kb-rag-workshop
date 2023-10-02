#!/bin/bash

# constants
BEDROCK_SDK_S3_PATH=https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.zip
BEDROCK_CONDA_ENV=bedrock_py39
BEDROCK_DIR=/home/ec2-user/SageMaker/bedrock
PY_VER=3.9

# create conda env
conda remove -n $BEDROCK_CONDA_ENV --all -y
conda create --name $BEDROCK_CONDA_ENV -y python=$PY_VER ipykernel
source activate $BEDROCK_CONDA_ENV

# create bedrock dir
rm -rf $BEDROCK_DIR
mkdir -p $BEDROCK_DIR

# download and install sdk
wget $BEDROCK_SDK_S3_PATH -P $BEDROCK_DIR
sdk_file_name=`basename $BEDROCK_SDK_S3_PATH`
unzip $BEDROCK_DIR/$sdk_file_name -d $BEDROCK_DIR

# all set to pip install the bedrock packages, these are awscli, boto3 and botocore
pip install $BEDROCK_DIR/*.whl
pip install langchain==0.0.297
pip install opensearch-py==2.3.1

echo "all done"
