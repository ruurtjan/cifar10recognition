from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Input
from sklearn.metrics import classification_report
import os
import pickle
import numpy as np
import logging

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

def main():
    """
        Train model on train data set and save model and quality of model to pickle files.
    """
    with open(os.path.join(os.getcwd(), "data", "processed", "data_dictionary.p"), "rb") as pickle_file:
        data_dict = pickle.load(pickle_file)

    model = conv_net(data_dict['train_X'], data_dict['train_y'], data_dict['val_X'], data_dict['val_y'],
                     len(data_dict['label_to_names']))
    model_quality_output(model, data_dict['test_X'], data_dict['test_y'], data_dict['label_to_names'])
    with open(os.path.join(os.getcwd(), "models", "model.p"), "wb") as pickle_file:
        pickle.dump(model, pickle_file)


def conv_net(train_X, train_y, val_X, val_y, nr_classes):
    """
    Trains model based on train_X, train_y data and validated by val_X, val_y data. A convolutional network
    is used to train the model.

    :param train_X: train feature data set
    :param train_y: train label data set
    :param val_X: validate feature data set
    :param val_y: validate label data set
    :param nr_classes: amount of classes in data
    :return model: fitted CNN model
    """
    # We definieren de input van het netwerk als de shape van de input,
    # minus de dimensie van het aantal plaatjes, uiteindelijk dus (3, 32, 32).
    input = Input(shape=train_X.shape[1:])

    # Eerste convolutielaag
    # Padding valid betekent dat we enkel volledige convoluties gebruiken, zonder padding
    # Data format channels_first betekent dat de channels eerst komen, en dan pas de size van ons plaatje
    # Dus (3, 32, 32) in plaats van (32, 32, 3)
    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='valid',
                  data_format='channels_first', activation='relu')(input)

    # Nog een convolutielaag, dit keer met stride=2 om de inputsize te verkleinen
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='valid',
                  data_format='channels_first', activation='relu', strides=(2, 2))(conv)

    # Voeg een flatten laag toe, om te schakelen naar de dense layer
    flatten = Flatten()(conv)

    # De softmax laag voor de probabilities
    output_layer = Dense(units=nr_classes, activation='softmax')(flatten)

    model = Model(inputs=input, outputs=output_layer)

    # Het model moet nog gecompiled worden en loss+learning functie gespecificeerd worden
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x=train_X, y=train_y, batch_size=50, epochs=10, validation_data=(val_X, val_y), verbose=2)
    return model


def model_quality_output(model, test_X, test_y, label_to_names):
    """
    Determines quality of model on test data set, prints and saves this to pickle file.

    :param model: model to predict labels of test_X
    :param test_X: test feature data set
    :param test_y: test label data set
    :param label_to_names: names of labels
    """
    predictions = np.array(model.predict(test_X, batch_size=100))
    test_y = np.array(test_y, dtype=np.int32)
    # Take the highest prediction
    predictions = np.argmax(predictions, axis=1)

    # Print resultaten
    accuracy = np.sum(predictions == test_y) / float(len(predictions))
    logger.info("Accuracy = {}".format(accuracy))
    with open(os.path.join(os.getcwd(), "models", "accuracy.p"), "wb") as pickle_file:
        pickle.dump(accuracy, pickle_file)

    report = classification_report(test_y, predictions, target_names=list(label_to_names.values()))
    logger.info(report)
    with open(os.path.join(os.getcwd(), "models", "report.p"), "wb") as pickle_file:
        pickle.dump(report, pickle_file)


if __name__ == '__main__':
    main()
