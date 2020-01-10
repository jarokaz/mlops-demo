FROM tensorflow/tfx:0.15.0
SHELL ["/bin/bash", "-c"]
WORKDIR /
RUN wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-275.0.0-linux-x86_64.tar.gz \
&& tar xvf google-cloud-sdk-275.0.0-linux-x86_64.tar.gz \
&& ./google-cloud-sdk/install.sh -q 
ENV PATH="/google-cloud-sdk/bin:${PATH}" 
RUN RELEASE=0.1.36 \
&& pip install google-resumable-media==0.4.1 https://storage.googleapis.com/ml-pipeline/release/$RELEASE/kfp.tar.gz 

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

ENTRYPOINT ["tfx"]
