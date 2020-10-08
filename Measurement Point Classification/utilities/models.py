from tensorflow.data import Dataset
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, MaxPool1D, Flatten, AveragePooling1D
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


@tf.function
def label_func(data, label):
    """
    Transforms the labels to be compatible with the double output of 
    "dnn_split" pooling
    """
    shot_labels = tf.repeat(tf.expand_dims(label, 0), 64, axis=0)
    MP_label = label
    return data, {'shot_output': shot_labels, 'MP_output': MP_label}


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


def NN_0_shot(input_shape, out_shape):
    """
    Creates and compiles the required Neural Network.
    Fully-Connected Neural Network with 0 hidden layers
    It is supposed to take as input a single shot; i.e. a single array spectrum
    """
    model_input = model = Input(shape=(input_shape,))
    model = Dense(out_shape, activation='softmax')(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def NN_1_shot(input_shape, out_shape):
    """
    Creates and compiles the required Neural Network.
    Fully-Connected Neural Network with 1 hidden layer
    It is supposed to take as input a single shot; i.e. a single array spectrum
    """
    model_input = model = Input(shape=(input_shape,))
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(out_shape, activation='softmax')(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def NN_3_shot(input_shape, out_shape):
    """
    Creates and compiles the required Neural Network.
    Fully-Connected Neural Network with 3 hidden layers
    It is supposed to take as input a single shot; i.e. a single array spectrum
    """
    model_input = model = Input(shape=(input_shape,))
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(128, activation='relu', name='last_hidden_layer')(model)
    model = Dropout(0.01)(model)
    model = Dense(out_shape, activation='softmax')(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def NN_5_shot(input_shape, out_shape):
    """
    Creates and compiles the required Neural Network.
    Fully-Connected Neural Network with 5 hidden layers
    It is supposed to take as input a single shot; i.e. a single array spectrum
    """
    model_input = model = Input(shape=(input_shape,))
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(128, activation='relu', name='last_hidden_layer')(model)
    model = Dropout(0.01)(model)
    model = Dense(out_shape, activation='softmax')(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def NN_0_MP(input_shape, out_shape, pooling='average', n_shots=64):
    """
    Creates and compiles the required Neural Network.
    Fully-Connected Neural Network with 0 hidden layers followed by a pooling operation.
        "dnn" pooling and "dnn_split" pooling both indicate a fully-connected layer for pooling
        however "dnn_split" differs by adding a second output right before the pooling operation.
        This second output is used to push the network towards having correct predictions on each
        shot rather than only the entire MP.
    It is supposed to take as input an entire Measurement Point or just the central grid;
        i.e. an 8x8 or 4x4 grid of shots (represented as a matrix with 64 or 16 lines where each line is a shot)
    """
    model_input = model = Input(shape=(n_shots, input_shape))
    shot_output = model = Dense(
        out_shape, activation='softmax', name='shot_output')(model)
    if pooling == 'average':
        model = AveragePooling1D(pool_size=n_shots)(model)
        model = Flatten()(model)
    elif pooling == 'max':
        model = MaxPool1D(pool_size=n_shots,
                          data_format='channels_last')(model)
        model = Flatten()(model)
    else:
        model = Flatten()(model)
        model = Dense(out_shape, activation='softmax', name='MP_output')(model)
        if pooling == 'dnn_split':
            losses = {"shot_output": 'categorical_crossentropy',
                      "MP_output": 'categorical_crossentropy'}
            lossWeights = {"shot_output": 1.0, "MP_output": 1.0}
            model = Model(inputs=model_input, outputs=[shot_output, model])
            opt_m1 = Adam(lr=0.0001)
            model.compile(optimizer=opt_m1, loss=losses,
                          loss_weights=lossWeights, metrics=['accuracy'])
            return model
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def NN_1_MP(input_shape, out_shape, pooling='average', n_shots=64):
    """
    Creates and compiles the required Neural Network.
    Fully-Connected Neural Network with 1 hidden layer followed by a pooling operation.
        "dnn" pooling and "dnn_split" pooling both indicate a fully-connected layer for pooling
        however "dnn_split" differs by adding a second output right before the pooling operation.
        This second output is used to push the network towards having correct predictions on each
        shot rather than only the entire MP.
    It is supposed to take as input an entire Measurement Point or just the central grid;
        i.e. an 8x8 or 4x4 grid of shots (represented as a matrix with 64 or 16 lines where each line is a shot)
    """
    model_input = model = Input(shape=(n_shots, input_shape))
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.01)(model)
    shot_output = model = Dense(
        out_shape, activation='softmax', name='shot_output')(model)
    if pooling == 'average':
        model = AveragePooling1D(pool_size=n_shots)(model)
        model = Flatten()(model)
    elif pooling == 'max':
        model = MaxPool1D(pool_size=n_shots,
                          data_format='channels_last')(model)
        model = Flatten()(model)
    else:
        model = Flatten()(model)
        model = Dense(out_shape, activation='softmax', name='MP_output')(model)
        if pooling == 'dnn_split':
            model = Model(inputs=model_input, outputs=[shot_output, model])

            opt_m1 = Adam(lr=0.0001)
            losses = {"shot_output": 'categorical_crossentropy',
                      "MP_output": 'categorical_crossentropy'}
            lossWeights = {"shot_output": 1.0, "MP_output": 1.0}

            model.compile(optimizer=opt_m1, loss=losses,
                          loss_weights=lossWeights, metrics=['accuracy'])
            return model
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def NN_3_MP(input_shape, out_shape, pooling='average', n_shots=64):
    """
    Creates and compiles the required Neural Network.
    Fully-Connected Neural Network with 3 hidden layer followed by a pooling operation.
        "dnn" pooling and "dnn_split" pooling both indicate a fully-connected layer for pooling
        however "dnn_split" differs by adding a second output right before the pooling operation.
        This second output is used to push the network towards having correct predictions on each
        shot rather than only the entire MP.
    It is supposed to take as input an entire Measurement Point or just the central grid;
        i.e. an 8x8 or 4x4 grid of shots (represented as a matrix with 64 or 16 lines where each line is a shot)
    """
    model_input = model = Input(shape=(n_shots, input_shape))
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(128, activation='relu', name='last_hidden_layer')(model)
    model = Dropout(0.01)(model)
    shot_output = model = Dense(
        out_shape, activation='softmax', name='shot_output')(model)
    if pooling == 'average':
        model = AveragePooling1D(pool_size=n_shots)(model)
        model = Flatten()(model)
    elif pooling == 'max':
        model = MaxPool1D(pool_size=n_shots,
                          data_format='channels_last')(model)
        model = Flatten()(model)
    else:
        model = Flatten()(model)
        model = Dense(out_shape, activation='softmax', name='MP_output')(model)
        if pooling == 'dnn_split':
            model = Model(inputs=model_input, outputs=[shot_output, model])

            opt_m1 = Adam(lr=0.0001)
            losses = {"shot_output": 'categorical_crossentropy',
                      "MP_output": 'categorical_crossentropy'}
            lossWeights = {"shot_output": 1.0, "MP_output": 1.0}

            model.compile(optimizer=opt_m1, loss=losses,
                          loss_weights=lossWeights, metrics=['accuracy'])
            return model
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def NN_5_MP(input_shape, out_shape, pooling='average', n_shots=64):
    """
    Creates and compiles the required Neural Network.
    Fully-Connected Neural Network with 5 hidden layer followed by a pooling operation.
        "dnn" pooling and "dnn_split" pooling both indicate a fully-connected layer for pooling
        however "dnn_split" differs by adding a second output right before the pooling operation.
        This second output is used to push the network towards having correct predictions on each
        shot rather than only the entire MP.
    It is supposed to take as input an entire Measurement Point or just the central grid;
        i.e. an 8x8 or 4x4 grid of shots (represented as a matrix with 64 or 16 lines where each line is a shot)
    """
    model_input = model = Input(shape=(n_shots, input_shape))
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.01)(model)
    model = Dense(128, activation='relu', name='last_hidden_layer')(model)
    model = Dropout(0.01)(model)
    shot_output = model = Dense(
        out_shape, activation='softmax', name='shot_output')(model)
    if pooling == 'average':
        model = AveragePooling1D(pool_size=n_shots)(model)
        model = Flatten()(model)
    elif pooling == 'max':
        model = MaxPool1D(pool_size=n_shots,
                          data_format='channels_last')(model)
        model = Flatten()(model)
    else:
        model = Flatten()(model)
        model = Dense(out_shape, activation='softmax', name='MP_output')(model)
        if pooling == 'dnn_split':
            model = Model(inputs=model_input, outputs=[shot_output, model])

            opt_m1 = Adam(lr=0.0001)
            losses = {"shot_output": 'categorical_crossentropy',
                      "MP_output": 'categorical_crossentropy'}
            lossWeights = {"shot_output": 1.0, "MP_output": 1.0}

            model.compile(optimizer=opt_m1, loss=losses,
                          loss_weights=lossWeights, metrics=['accuracy'])
            return model
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.0001)
    model.compile(optimizer=opt_m1,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_run_and_evaluate_shot(model_number, mode, output_type,
                                train_data, train_y,
                                test_data, test_y):
    """
    This function will create and train the model corresponding to the given model number.
    Notice that major vote pooling (which isn't really pooling) is done via this function instead
    of it's MP counterpart below
    """
    model_funcs = [NN_0_shot,
                   NN_1_shot,
                   NN_3_shot,
                   NN_5_shot]

    input_shape = train_data.shape[-1]
    out_shape = train_y.shape[1]

    keras.backend.clear_session()

    train_inputs = Dataset.from_tensor_slices(train_data)
    train_outputs = Dataset.from_tensor_slices(train_y)
    train_dataset = Dataset.zip((train_inputs, train_outputs))
    train_dataset = train_dataset.shuffle(64000).batch(
        256).prefetch(tf.data.experimental.AUTOTUNE)

    model = model_funcs[model_number](input_shape, out_shape)
    _ = model.fit(train_dataset, epochs=150, verbose=0)

    # COMPUTING ACCURACY SCORES
    if mode == 'standard':
        if output_type == 'accuracies':
            output = model.evaluate(test_data, test_y, verbose=0)[1]
        else:
            output = [model.predict(test_data), test_y[0]]
    elif mode == 'major_vote':
        output = evaluate_major_vote(model, test_data, test_y, output_type)
    else:
        output = compute_model_heatmap(model, test_data, test_y)

    return output


def model_run_and_evaluate_MP(model_number, pooling, output_type,
                              train_data, train_y,
                              test_data, test_y):

    model_funcs = [NN_0_MP,
                   NN_1_MP,
                   NN_3_MP,
                   NN_5_MP]

    input_shape = train_data.shape[-1]
    n_shots = train_data.shape[1]
    out_shape = train_y.shape[1]

    train_inputs = Dataset.from_tensor_slices(train_data)
    train_outputs = Dataset.from_tensor_slices(train_y)
    train_dataset = Dataset.zip((train_inputs, train_outputs))
    if pooling == 'dnn_split':
        train_dataset = train_dataset.map(
            label_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(10000).batch(
        32).prefetch(tf.data.experimental.AUTOTUNE)

    test_inputs = Dataset.from_tensor_slices(test_data)
    test_outputs = Dataset.from_tensor_slices(test_y)
    test_dataset = Dataset.zip((test_inputs, test_outputs))
    if pooling == 'dnn_split':
        test_dataset = test_dataset.map(
            label_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.shuffle(10000).batch(
        32).prefetch(tf.data.experimental.AUTOTUNE)

    model = model_funcs[model_number](
        input_shape, out_shape, pooling, n_shots=n_shots)
    _ = model.fit(train_dataset, epochs=150, verbose=0)

    if output_type == 'predictions':
        predictions = model.predict(test_dataset)
        if pooling == 'dnn_split':
            output = [predictions[1], test_y[0]]
        else:
            output = [predictions, test_y[0]]
    elif output_type == 'accuracies':
        if pooling != 'dnn_split':
            output = model.evaluate(test_dataset, verbose=0)[1]
        else:
            output = model.evaluate(test_dataset, verbose=0)[-1]
    elif output_type == 'pooling_weights':
        weights = model.get_layer('MP_output').get_weights()[0]
        output = np.zeros(shape=(n_shots, out_shape**2))
        for i in range(out_shape):
            output[i] = weights[i*out_shape:(i+1)*out_shape].flatten()
        output = np.linalg.norm(output, axis=1)

    keras.backend.clear_session()

    return output


def model_run_and_return_history_MP(model_number, pooling,
                                    train_data, train_y,
                                    test_data, test_y):
    model_funcs = [NN_0_MP,
                   NN_1_MP,
                   NN_3_MP,
                   NN_5_MP]

    input_shape = train_data.shape[-1]
    out_shape = train_y.shape[1]

    train_inputs = Dataset.from_tensor_slices(train_data)
    train_outputs = Dataset.from_tensor_slices(train_y)
    train_dataset = Dataset.zip((train_inputs, train_outputs))
    if pooling == 'dnn_split':
        train_dataset = train_dataset.map(
            label_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    test_inputs = Dataset.from_tensor_slices(test_data)
    test_outputs = Dataset.from_tensor_slices(test_y)
    test_dataset = Dataset.zip((test_inputs, test_outputs))
    if pooling == 'dnn_split':
        test_dataset = test_dataset.map(
            label_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.shuffle(1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    model = model_funcs[model_number](input_shape, out_shape, pooling)
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=300, verbose=0)

    if pooling != 'dnn_split':
        output_history = [history.history['accuracy'],
                          history.history['val_accuracy']]
    else:
        output_history = [history.history['MP_output_accuracy'],
                          history.history['val_MP_output_accuracy']]
    keras.backend.clear_session()

    return output_history


def evaluate_major_vote(model, test_data, test_y, output_type):
    correct_class = np.argmax(test_y[0])

    major_vote_results = []
    predictions = []
    for MP in test_data:
        MP_predictions = []
        for shot in MP:
            shot_prediction = model.predict(np.array([shot]))
            MP_predictions.append(np.argmax(shot_prediction))

        uniques, counts = np.unique(MP_predictions, return_counts=True)
        major_vote = uniques[np.argmax(counts)]
        predictions.append(major_vote)
        major_vote_results.append((major_vote == correct_class)*1)

    if output_type == 'accuracies':
        output = np.mean(major_vote_results)
    else:
        output = [np.array(predictions), test_y[0]]

    return output


def compute_model_heatmap(model, test_data, test_y):
    """
    Args:
            model: a shot-level Keras model
            test_data: np.array, input data to test the model on
            test_y: np.array, corresponding output labels
    Returns:
            heatmap: np.array of shape (64,) corresponding to the 8x8 grid
                    of mean accuracies computed for each shot in the 8x8 grid over
                    the given test dataset
    """
    correct_class = np.argmax(test_y[0])

    grid_points_probs = [[] for _ in range(64)]
    for MP in test_data:
        for index, shot in enumerate(MP):
            shot_prediction = model.predict(np.array([shot]))
            correct_class_prob = shot_prediction[0][correct_class]

            grid_points_probs[index].append(correct_class_prob)

    grid_points_probs = np.array(grid_points_probs)

    heatmap = np.mean(grid_points_probs, axis=1)

    return heatmap
