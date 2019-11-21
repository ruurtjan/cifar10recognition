import os
import pickle
from logging.config import dictConfig

import numpy as np
from flask import Flask, request

from features.build_features import normalize

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '%(message)s',
    }},
    'handlers': {'console': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
})

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    request.get_data()
    data = str(request.data).split(',\\n')
    data = np.array([float(line.replace(' ', '').replace('\\n', '').replace('[', '').replace(']', '')
                           .replace('b', '').replace("'", '')) for line in data]).reshape(-1, 3, 32, 32).astype(
        "float32")
    image = normalize(data, extra_info_dict['mean'], extra_info_dict['std'])
    return str(extra_info_dict['label_to_names'][np.argmax(model.predict(image))])


if __name__ == '__main__':
    with open(os.path.join(os.getcwd(), "models", "model.pickle"), "rb") as pickle_file:
        model = pickle.load(pickle_file)

    with open(os.path.join(os.getcwd(), "models", "extra_info_dictionary.pickle"), "rb") as pickle_file:
        extra_info_dict = pickle.load(pickle_file)

    app.run(host='0.0.0.0', threaded=False)
