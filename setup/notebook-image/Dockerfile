FROM gcr.io/deeplearning-platform-release/tf-cpu.1-15
RUN apt-get update -y && apt-get -y install kubectl
RUN pip install -U urllib3==1.24.2 google-resumable-media==0.5.0 six==1.12 apache-beam==2.16 pyarrow==0.14.0 tfx-bsl==0.15.1  \
&& pip install -U tfx==0.15 \
&& RELEASE=0.1.36 \
&& pip install https://storage.googleapis.com/ml-pipeline/release/$RELEASE/kfp.tar.gz

