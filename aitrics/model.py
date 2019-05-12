import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, CuDNNLSTM, Dropout, BatchNormalization

def init_model(args):
    # whether to use modified dataset
    if args.use_modified:
        sequence_length = 5
    else:
        sequence_length = 4

    # construct model
    if args.model == 'mlp':
        model = Sequential()

        model.add(Dense(50, input_dim=sequence_length*34))
        model.add(Activation('relu'))
        
        if args.dropout:
            model.add(Dropout(0.8))

        model.add(Dense(30))
        model.add(Activation('relu'))

        if args.dropout:
            model.add(Dropout(0.8))

        model.add(Dense(1, activation='sigmoid'))

        # early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_auc',
                                                min_delta=0.0001,
                                                patience=100,
                                                verbose=1,
                                                mode='max',
                                                baseline=None,
                                                restore_best_weights=True)

    elif args.model == 'lstm':
        model = Sequential()

        model.add(CuDNNLSTM(units=16, input_shape=(sequence_length, 34),
                       return_sequences=True))

        if args.dropout:
            model.add(Dropout(0.8))

        model.add(CuDNNLSTM(units=16,
                       return_sequences=False))

        model.add(Dense(1, activation='sigmoid'))

        # early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_auc',
                                                min_delta=0.0001,
                                                patience=30,
                                                verbose=1,
                                                mode='max',
                                                baseline=None,
                                                restore_best_weights=True)

    model.summary()

    return model, early_stopping_callback
