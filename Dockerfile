FROM tensorflow/tensorflow:latest-py3
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt
COPY models/model.pickle /usr/src/
COPY models/extra_info_dictionary.pickle /usr/src/
COPY src /usr/src
EXPOSE 5000
CMD ["python", "/usr/src/models/api.py"]