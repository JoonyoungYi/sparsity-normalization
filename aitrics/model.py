import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, CuDNNLSTM, Dropout, BatchNormalization
from .LayerNormLSTM import LSTM

def init_model(args):
    # whether to use modified dataset
    if args.use_modified:
        sequence_length = 5
    else:
        sequence_length = 4

    # construct model
    if args.model == 'mlp':
        model = Sequential()

        model.add(Dense(100, input_dim=sequence_length*34))

        if args.batch_norm_mlp:
            model.add(BatchNormalization())

        model.add(Activation('relu'))
        
        if args.dropout:
            model.add(Dropout(0.5))

        model.add(Dense(100))

        if args.batch_norm_mlp:
            model.add(BatchNormalization())

        model.add(Activation('relu'))

        if args.dropout:
            model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))

        # early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_auc',
                                                min_delta=0.0001,
                                                patience=args.epochs_early_stop,
                                                verbose=1,
                                                mode='max',
                                                baseline=None,
                                                restore_best_weights=True)

    elif args.model == 'lstm':
        model = Sequential()

        if args.layer_norm_lstm:
            model.add(LSTM(units=100, input_shape=(sequence_length, 34), layer_norm=True,
                           return_sequences=True))
        else:
            model.add(CuDNNLSTM(units=100, input_shape=(sequence_length, 34),
                           return_sequences=True))

        if args.dropout:
            model.add(Dropout(0.5))

        if args.layer_norm_lstm:
            model.add(LSTM(units=100, layer_norm=True,
                           return_sequences=False))
        else:
            model.add(CuDNNLSTM(units=100,
                           return_sequences=False))

        model.add(Dense(1, activation='sigmoid'))

        # early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_auc',
                                                min_delta=0.0001,
                                                patience=args.epochs_early_stop,
                                                verbose=1,
                                                mode='max',
                                                baseline=None,
                                                restore_best_weights=True)

    model.summary()

    return model, early_stopping_callback
