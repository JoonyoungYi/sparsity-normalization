import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, CuDNNLSTM

'''Requirements of init_model
args.use_modified는 True/False
args.model은 mlp/lstm
args.epochs
'''

'''Output of init_model
model, lr_reducer
'''

def init_model(args):
    # whether to use modified dataset
    if args.use_modified:
        sequence_length = 5
    else:
        sequence_length = 4

    # construct model
    if args.model == 'mlp':
        model = Sequential()

        model.add(Dense(128, input_dim=sequence_length*34))
        model.add(Activation('tanh'))

        model.add(Dense(64))
        model.add(Activation('tanh'))

        model.add(Dense(1, activation='sigmoid'))

        # early stopping callback
        lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_auc',
                                                       factor=0.2,
                                                       patience=10,
                                                       verbose=0,
                                                       mode='max',
                                                       min_delta=0.0001,
                                                       cooldown=0,
                                                       min_lr=5e-6)

    elif args.model == 'lstm':
        model = Sequential()

        model.add(CuDNNLSTM(units=300, input_shape=(sequence_length, 34),
                       return_sequences=True))

        model.add(CuDNNLSTM(units=100,
                       return_sequences=False))

        model.add(Activation('tanh'))

        model.add(Dense(50, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))

        # early stopping callback
        lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_auc',
                                                       factor=0.8,
                                                       patience=max(10,round(args.epochs/10)),
                                                       verbose=0,
                                                       mode='max',
                                                       min_delta=0.00001,
                                                       cooldown=0,
                                                       min_lr=5e-6)

    model.summary()

    return model, lr_reducer
