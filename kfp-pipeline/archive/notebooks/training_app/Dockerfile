
FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire cloudml-hypertune
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
