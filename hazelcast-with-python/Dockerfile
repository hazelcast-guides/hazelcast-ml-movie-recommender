FROM hazelcast/hazelcast:5.2.1
USER root

# Install python/pip see: https://stackoverflow.com/questions/62554991/how-do-i-install-python-on-alpine-linux
ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools

# copy python into image
RUN mkdir -p python-recommender; chown -R hazelcast python-recommender
COPY python-recommender .

USER hazelcast
