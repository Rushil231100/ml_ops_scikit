FROM ubuntu:18.04
COPY docker /exp/docker
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
RUN mkdir /exp/models
WORKDIR /exp/docker
EXPOSE 5001
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
