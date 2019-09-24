import time
import csv
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from io import BytesIO
import os

os.environ['TF2_BEHAVIOR'] = '1'  # needed for keras model serializing

def train_nn(train, categories):
    ## split data set
    X_train, X_test, Y_train, Y_test = train_test_split(train, categories, test_size=0.33, random_state=42, stratify=categories)
    
    ## max min scalar on parameters
    X_scaler = MinMaxScaler(feature_range=(0,1))
     
    ## Preprocessing the dataset
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.fit_transform(X_test)
     
    ## One hot encode Y
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_train_enc = onehot_encoder.fit_transform(np.array(Y_train).reshape(-1,1))
    Y_test_enc = onehot_encoder.fit_transform(np.array(Y_test).reshape(-1,1))
    # api.logger.info(str(Y_test_enc))

    # prepare dataset for training epochs
    dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, Y_train_enc))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test_scaled, Y_test_enc))
    dataset_test = dataset_test.shuffle(1000)
    dataset_test = dataset_test.batch(32)
    dataset_test = dataset_test.repeat()
    
    # MLP network setup
    model = tf.keras.Sequential([tf.keras.layers.Dense(16, input_dim=4), tf.keras.layers.Dense(3, activation=tf.nn.softmax)])

    # opt = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    
    history = model.fit(dataset, steps_per_epoch=32, epochs=100, verbose=1)
    # api.logger.info(str(history.history))
    loss, accuracy = model.evaluate(dataset_test, steps=32)
    api.logger.info("loss:%f" % (loss))
    api.logger.info("accuracy: %f" % (accuracy))
    
    return model, accuracy

def process_train_data(data):
    api.logger.info(data)
    reader = csv.reader(data.split('\n'), delimiter=',')
    train = []
    categories = []
    for row in list(reader)[1:]:  # skip header
        data = row[:4]
        category = row[4]

        train.append(data)
        categories.append(category)

    api.logger.info(str(list(categories)))

    return train, categories

def on_input(data):
    train, categories = process_train_data(data)
    model, accuracy =  train_nn(train, categories)
    # to send metrics to the Submit Metrics operator, create a Python dictionary of key-value pairs
    metrics_dict = {"accuracy": str(accuracy)}
    # send the metrics to the output port - Submit Metrics operator will use this to persist the metrics 
    api.send("metrics", api.Message(metrics_dict))

    # create & send the model blob to the output port - Artifact Producer operator will use this to persist the model and create an artifact ID
    model_h5 = BytesIO()
    model.save(model_h5)
    api.send("modelBlob", model_h5.getvalue())
    model_h5.close()

api.set_port_callback("input", on_input)
