#!/bin/bash
# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Submits a Cloud Build job that builds and deploys
# the pipelines and pipelines components 

SUBSTITUTIONS=\
_PIPELINE_NAME=onlinenews-predictor,\
_TFX_IMAGE_NAME=lab-14-tfx-image,\
_GCP_REGION=us-central1,\
_ARTIFACT_STORE_URI=gs://mlops-workshop-dev-artifact-store,\
_DATA_ROOT_URI=gs://mlops-workshop-dev-artifact-store/lab-datasets/online_news,\
_PIPELINE_FOLDER=lab-14-tfx-caip-e2e/pipeline-dsl,\
_PIPELINE_DSL=pipeline_dsl.py,\
_KFP_INVERSE_PROXY_HOST=40a6ff3f4b68ff12-dot-us-central1.notebooks.googleusercontent.com,\
_PYTHON_VERSION=3.7,\
_RUNTIME_VERSION=1.15,\
TAG_NAME=latest

gcloud builds submit ../.. --config cloudbuild.yaml --substitutions $SUBSTITUTIONS


