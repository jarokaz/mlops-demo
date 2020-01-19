FROM gcr.io/deeplearning-platform-release/tf-cpu.1-15
RUN apt-get update -y && apt-get -y install kubectl
RUN cd /usr/local/bin \
&& wget https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv3.3.0/kustomize_v3.3.0_linux_amd64.tar.gz \
&& tar xvf kustomize_v3.3.0_linux_amd64.tar.gz \
&& rm kustomize_v3.3.0_linux_amd64.tar.gz
RUN pip install -U six==1.12 apache-beam==2.16 pyarrow==0.14.0 tfx-bsl==0.15.1 \
&& pip install -U tfx==0.15 \
&& RELEASE=0.1.36 \
&& pip install https://storage.googleapis.com/ml-pipeline/release/$RELEASE/kfp.tar.gz

