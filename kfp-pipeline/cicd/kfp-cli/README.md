This folder contains the Dockerfile and the build script for the Cloud Build custom builder that encapsulates KFP CLI.

There is a [bug](https://github.com/kubeflow/pipelines/issues/2764) in the 1.37 and 1.38 versions of KFP SDK that causes the `kfp pipeline upload` command to fail. As a temporary mitigation we use the KFP SDK version 1.36 in the builder.
