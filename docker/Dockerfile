FROM ubuntu:18.04
COPY ml_ops_scikit /exp/ml_ops_scikit
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
RUN mkdir /exp/models
WORKDIR /exp
CMD ["python3", "./ml_ops_scikit/plot_digits_classification.py"]
