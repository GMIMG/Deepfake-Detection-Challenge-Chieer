FROM python:3.6
ADD . /www
COPY requirements.txt /tmp/
RUN pip install uwsgi
RUN pip install -r /tmp/requirements.txt && \
	rm /tmp/requirements.txt
COPY factory.py /usr/local/lib/python3.6/site-packages/mtcnn/network/
WORKDIR /www
CMD uwsgi uwsgi.ini