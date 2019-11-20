import os
import pickle

import numpy as np
from flask import Flask, request

from features.build_features import normalize

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    with open(os.path.join(os.getcwd(), "models", "model.pickle"), "rb") as pickle_file:
        model = pickle.load(pickle_file)

    with open(os.path.join(os.getcwd(), "models", "extra_info_dictionary.pickle"), "rb") as pickle_file:
        extra_info_dict = pickle.load(pickle_file)

    request.get_data()
    data = str(request.data).split(',\\n')
    data = np.array([float(line.replace(' ', '').replace('\\n', '').replace('[', '').replace(']', '')
                           .replace('b', '').replace("'", '')) for line in data]).reshape(-1, 3, 32, 32).astype(
        "float32")
    image = normalize(data, extra_info_dict['mean'], extra_info_dict['std'])
    return str(extra_info_dict['label_to_names'][np.argmax(model.predict(image))])


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=False)
