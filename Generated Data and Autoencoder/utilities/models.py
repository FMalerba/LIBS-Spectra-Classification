from tensorflow.data import Dataset
import xgboost as xgb
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, MaxPool1D, Flatten, AveragePooling1D
from tensorflow.keras.models import Model
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def get_autoencoder_model(input_shape, embedding_shape=128, sigmoid=False):
    model_input = model = Input(shape=(input_shape, ))
    #model = Dense(1024, activation='relu')(model)
    #model = Dense(512, activation='relu')(model)
    #model = Dense(128, activation='relu')(model)
    model = Dense(embedding_shape, activation='relu', name='embedding')(model)
    #model = Dense(128, activation='relu')(model)
    #model = Dense(512, activation='relu')(model)
    #model = Dense(1024, activation='relu')(model)
    if sigmoid:
        model = Dense(input_shape, activation='sigmoid')(model)
    else:
        model = Dense(input_shape, activation='linear')(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.001)
    model.compile(optimizer=opt_m1, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Tensorflow accuracy metric is inaccurate in the case of a multi-hot output
def my_accuracy_metric(y_true, y_pred):
    """
    Args:
        y_true: multi-hot vectors of true y. Shape (n_samples, n_outputs)
        y_pred: prediction vector (to be rounded in order to get the multi-hot predictions). Same shape as y_true
    Returns:
        Accuracy corresponding to the actual number of entries that are correct. So for example a single prediction of 
        [0.8, 0.8, 0.8] with a ground truth of [1,0,1] would get rounded and then return an accuracy of 0.67 since 2
        out of 3 entries are correct
    """
    return tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.round(y_pred), y_true), dtype=tf.float32))


def NN_0_shot(input_shape, out_shape, classify_elements=False):
    """
    Creates and compiles the required Neural Network.
    If classify_elements = True then the output is going to be a multi-hot vector instead of a one-hot.
    In this case losses, metrics, and activation function for the last layer have to be changed appropriately.
    Fully-Connected Neural Network with 0 hidden layers
    It is supposed to take as input a single shot; i.e. a single array spectrum
    """
    model_loss = tf.keras.losses.binary_crossentropy if classify_elements else 'categorical_crossentropy'
    model_output_activation = 'sigmoid' if classify_elements else 'softmax'
    model_metric = my_accuracy_metric if classify_elements else 'accuracy'
    model_input = model = Input(shape=(input_shape,))
    model = Dense(out_shape, activation=model_output_activation)(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1, loss=model_loss, metrics=[model_metric])
    return model


def NN_1_shot(input_shape, out_shape, classify_elements=False):
    """
    Creates and compiles the required Neural Network.
    If classify_elements = True then the output is going to be a multi-hot vector instead of a one-hot.
    In this case losses, metrics, and activation function for the last layer have to be changed appropriately.
    Fully-Connected Neural Network with 1 hidden layer
    It is supposed to take as input a single shot; i.e. a single array spectrum
    """
    model_loss = tf.keras.losses.binary_crossentropy if classify_elements else 'categorical_crossentropy'
    model_output_activation = 'sigmoid' if classify_elements else 'softmax'
    model_metric = my_accuracy_metric if classify_elements else 'accuracy'
    model_input = model = Input(shape=(input_shape,))
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(out_shape, activation=model_output_activation)(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1, loss=model_loss, metrics=[model_metric])
    return model


def NN_3_shot(input_shape, out_shape, classify_elements=False):
    """
    Creates and compiles the required Neural Network.
    If classify_elements = True then the output is going to be a multi-hot vector instead of a one-hot.
    In this case losses, metrics, and activation function for the last layer have to be changed appropriately.
    Fully-Connected Neural Network with 3 hidden layers
    It is supposed to take as input a single shot; i.e. a single array spectrum
    """
    model_loss = tf.keras.losses.binary_crossentropy if classify_elements else 'categorical_crossentropy'
    model_output_activation = 'sigmoid' if classify_elements else 'softmax'
    model_metric = my_accuracy_metric if classify_elements else 'accuracy'
    model_input = model = Input(shape=(input_shape,))
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(128, activation='relu', name='last_hidden_layer')(model)
    model = Dropout(0.01)(model)
    model = Dense(out_shape, activation=model_output_activation)(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1, loss=model_loss, metrics=[model_metric])
    return model


def model_run_and_evaluate(use_autoencoder, model_number, classify_elements, embedding_shape,
                           train_data, train_labels, train_y,
                           test_data, test_labels, test_y):
    model_funcs = [NN_0_shot,
                   NN_1_shot,
                   NN_3_shot,
                   xgb.XGBClassifier(max_depth=7, n_estimators=50,
                                     learning_rate=0.05, nthread=5),
                   xgb.XGBClassifier(max_depth=15, n_estimators=100, learning_rate=0.05, nthread=15)]

    if use_autoencoder:
        input_shape = train_data[0].shape[0]
        model = get_autoencoder_model(
            input_shape, embedding_shape=embedding_shape, sigmoid=True)
        _ = model.fit(train_data, train_data, epochs=10,
                      batch_size=256, verbose=0)
        autoencoder = Model(model.input, model.get_layer('embedding').output)
        train_data = autoencoder.predict(train_data)
        test_data = autoencoder.predict(test_data)

    input_shape = train_data.shape[1]
    out_shape = train_y.shape[1]

    # Neural Network Case
    if model_number <= 2:
        train_inputs = Dataset.from_tensor_slices(train_data)
        train_outputs = Dataset.from_tensor_slices(train_y)
        train_dataset = Dataset.zip((train_inputs, train_outputs))
        train_dataset = train_dataset.shuffle(100000).batch(
            256).prefetch(tf.data.experimental.AUTOTUNE)

        test_inputs = Dataset.from_tensor_slices(test_data)
        test_outputs = Dataset.from_tensor_slices(test_y)
        test_dataset = Dataset.zip((test_inputs, test_outputs))
        test_dataset = test_dataset.shuffle(20000).batch(
            32).prefetch(tf.data.experimental.AUTOTUNE)

        model = model_funcs[model_number](
            input_shape, out_shape, classify_elements)

        _ = model.fit(train_dataset, epochs=200, verbose=0)

        # COMPUTING ACCURACY SCORES
        model_acc = model.evaluate(test_dataset, verbose=0)[1]

    # Random Forest Case
    else:
        model = model_funcs[model_number]
        model.fit(train_data, train_labels)

        # COMPUTING ACCURACY SCORES
        preds = model.predict(test_data)
        model_acc = np.mean(preds == test_labels)

    keras.backend.clear_session()

    return model_acc


def element_classifier_accuracy(preds, true_y):
    """
    Args:
        preds: Prediction array for the different elements
        true_y: True labels for the different elements
    Returns:
        overall accuracy and accuracy for each element
    """
    preds = np.round(preds)
    overall_accuracy = np.sum(preds == true_y) / preds.size
    accuracy_per_element = np.mean(preds == true_y, axis=0)

    return overall_accuracy, accuracy_per_element
