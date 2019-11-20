FROM tensorflow/tensorflow:latest-py3
COPY requirements.txt /usr/src/
RUN pip install -r /usr/src/requirements.txt
ADD models /usr/models/
COPY src /usr/src
EXPOSE 5000
WORKDIR /usr
CMD ["python", "src/api.py"]
