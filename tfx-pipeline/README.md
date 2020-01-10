# Orchestrating model training and deployment with TFX and Cloud AI Platform

In this lab you will develop and operationalize a TFX pipeline that uses Kubeflow Pipelines for orchestration and Cloud Dataflow and Cloud AI Platform for data processing, training, and deployment:

1. In Exercise 1, you will review and understand the pipeline's source code - also referred to as the pipeline's DSL.

1. In Exercise 2, you will use **TFX CLI** to deploy the pipeline to KFP environment

1. In Exercise 3, you will use **TFX CLI** and **KFP UI** to submit and monitor pipeline runs.

1. In Exercise 4, you will author a **Cloud Build** CI/CD workflow that automates pipeline deployment.


## Lab scenario

You will be working with a variant of the [Online News Popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) dataset, which summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. The goal is to predict how popular the article will be on social networks. Specifically, in the original dataset the objective was to predict the number of times each article will be shared on social networks. In this variant, the goal is to predict the article's popularity percentile. For example, if the model predicts a score of 0.7, then it means it expects the article to be shared more than 70% of all articles.

The pipeline implements a typical TFX workflow as depicted on the below diagram:

![Lab 14 diagram](../images/lab-14-diagram.png).

The source data in a CSV file format is in the GCS bucket.

The TFX `ExampleGen`, `StatisticsGen`, `ExampleValidator`, `SchemaGen`, `Transform`, and `Evaluator` components use Cloud Dataflow as an execution engine. The `Trainer` and `Pusher` components use AI Platform Training and Prediction services.


## Lab setup

### AI Platform Notebook configuration
You will use the **AI Platform Notebooks** instance configured with a custom container image. To prepare the **AI Platform Notebooks** instance:

1. In **Cloud Shell**, navigate to the `lab-00-environment-setup/notebook-images/tf115-tfx015-kfp136` folder.
2. Build the container image
```
./build.sh
```
3. Provision the **AI Platform Notebook** instance based on a custom container image, following the  [instructions in AI Platform Notebooks Documentation](https://cloud.google.com/ai-platform/notebooks/docs/custom-container). In the **Docker container image** field, enter the following image name: `gcr.io/[YOUR_PROJECT_NAME]/tfx-kfp-dev:TF115-TFX015-KFP136`.

4. After the **AI Platform Notebooks** instance is ready, *open JupyterLab*.

5. Open a new terminal in **JupyterLab** and clone this repo under the `home` directory
```
cd /home
git clone https://github.com/jarokaz/mlops-labs.git
```

### Lab dataset
This lab uses the the [Online News Popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) dataset. The pipeline is designed to ingest the dataset from the GCS location inr in the *artifact store* bucket created during the environment setup  - `lab-00-environment-setup`. As you recall, the URI of the *artifact store* bucket created during the setup is `gs://[YOUR_PREFIX]-artifact-store`. To upload the *Online News Popularity* dataset, execute the following command from the *JupyterLab* terminal:

```
DATA_ROOT_URI=[YOUR_ARTIFACT_STORE_BUCKET_URI]/lab-datasets/online_news
gsutil cp gs://workshop-datasets/online_news/full/data.csv $DATA_ROOT_URI/data.csv 
```



## Lab Exercises
### Exercise 1  - Understanding the pipeline's DSL.

Follow the instructor who will walk you through the pipeline's DSL.

As described by the instructor, the pipeline in this lab uses a custom docker image that is a derivative of a base `tensorflow/tfx:0.15.0` image from [Docker Hub](https://hub.docker.com/r/tensorflow/tfx). The base `tfx` image includes TFX v0.15 and TensorFlow v2.0. The custom image modifies the base image by downgrading to TensorFlow to v1.15 and adding the `modules` folder with the `transform_train.py` file that contains data transformation and training code used by the pipeline's `Transform` and `Train` components.

The pipeline needs to use v1.15 of TensorFlow as AI Platform Prediction service, which is used as a deployment target, does not yet support v2.0 of TensorFlow.

### Exercise 2 - Deploying the pipeline
You can use **TFX CLI** to compile and deploy the pipeline to the KFP environment. As the pipeline uses the custom image, the first step is to build the image and push it to your project's **Container Registry**. You will use **Cloud Build** to build the image.

First, activate the `tfx` Python environment that hosts TFX and TFX CLI.
```
source activate tfx
```

To create the image, navigate to the `pipeline-dsl` folder and execute the following commands:
```
PROJECT_ID=[YOUR_PROJECT_ID]
IMAGE_NAME=lab-14-tfx-image
TAG=latest
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

gcloud builds submit --timeout 15m --tag ${IMAGE_URI} .
```

As explained by the instructor, the pipeline's DSL retrieves the settings controlling how the pipeline is compiled from the environment variables.To set the environment variables and compile and deploy the pipeline using  **TFX CLI**:

```
export PROJECT_ID=[YOUR_PROJECT_ID]
export ARTIFACT_STORE_URI=[YOUR_ARTIFACT_STORE_URI]
export DATA_ROOT_URI=[YOUR_DATA_ROOT_URI]
export TFX_IMAGE=[YOUR_TFX_IMAGE_URI]
export KFP_INVERSE_PROXY_HOST=[YOUR_INVERSE_PROXY_HOST]

export PIPELINE_NAME=online_news_model_training
export GCP_REGION=us-central1
export RUNTIME_VERSION=1.15
export PYTHON_VERSION=3.7


tfx pipeline create --pipeline_path pipeline_dsl.py --endpoint $KFP_INVERSE_PROXY_HOST
```

The instructor will walk you through the pipeline package file generated by the compiler and the pipeline deployment on KFP GKE cluster.

### Exercise 3 - Submitting and monitoring pipeline runs

After the pipeline has been deployed, you can trigger and monitor pipeline runs using **TFX CLI** and/or **KFP UI**.

To submit the pipeline run using **TFX CLI**:
```
tfx run create --pipeline_name online_news_model_training --endpoint $KFP_INVERSE_PROXY_HOST
```

To list all the active runs of the pipeline:
```
tfx run list --pipeline_name online_news_model_training --endpoint $KFP_INVERSE_PROXY_HOST
```

To retrieve the status of a given run:
```
tfx run status --pipeline_name online_news_model_training --run_id [YOUR_RUN_ID] --endpoint $KFP_INVERSE_PROXY_HOST
```
 To terminate a run:
 ```
 tfx run terminate --run_id [YOUR_RUN_ID] --endpoint $KFP_INVERSE_PROXY_HOST
 ```


### Exercise  4 - Authoring the CI/CD workflow that builds and deploy the KFP training pipeline

In this exercise you review and trigger a **Cloud Build** CI/CD workflow that automates the process of compiling and deploying the KTFX pipeline. The **Cloud Build** configuration uses both standard and custom [Cloud Build builders](https://cloud.google.com/cloud-build/docs/cloud-builders). The custom builder, which you build in the first part of the exercise, encapsulates **TFX CLI**. 

As of version 1.36 of **KFP** there is no support for pipeline versions. It will be added in future releases, with the intial functionality introduced in version 1.37. In the lab, you append the **Cloud Build** `$TAG_NAME` default substitution to the name of the pipeline to designate a pipeline version. When the pipeline versioning features is exposed through **KFP SDK** this exercise will be updated to use the feature.

1. Create a **Cloud Build** custom builder that encapsulates TFX CLI.
```
cd cicd/tfx-cli
./build.sh
```
2. Follow the instructor who will walk you through  the Cloud Build configuration in:
```
cicd/cloudbuild.yaml
```
3. Update the `build_pipeline.sh` script in the `cicd` folder with your KFP inverting proxy host.

4. Manually trigger the CI/CD build:
```
./build_pipeline.sh
```
