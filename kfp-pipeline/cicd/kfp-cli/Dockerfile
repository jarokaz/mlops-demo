FROM gcr.io/deeplearning-platform-release/base-cpu
SHELL ["/bin/bash", "-c"]
RUN apt-get update -y && apt-get -y install kubectl
RUN RELEASE=0.1.36 \
&& pip install https://storage.googleapis.com/ml-pipeline/release/$RELEASE/kfp.tar.gz 

ENTRYPOINT ["/bin/bash"]
