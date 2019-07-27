import keras
from keras.models import Sequential
from keras.layers import Layer, Dense, Activation, CuDNNLSTM, Dropout, GaussianDropout, BatchNormalization, Conv1D, Flatten
from .LayerNormLSTM import LSTM

class LayerNormalization(Layer):
    """Layer normalization for mlp model"""
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer='ones',
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer='zeros',
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        m = K.mean(layer, axis=-1, keepdims=True)
        std = K.std(layer, axis=-1, keepdims=True)
        layer_normed = (layer - m) / (std + K.epsilon())
        return self.gamma * layer_normed + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

def init_model(args):
    """
    Input:
    args.model
    args.batch_norm_mlp
    args.layer_norm_mlp
    args.layer_norm_lstm
    args.dropout
    args.epochs_early_stop

    Output:
    model
    early_stopping_callback
    """
    sequence_length = 4

    # construct mlp model
    if args.model == 'mlp':
        model = Sequential()

        model.add(Dense(50, input_dim=sequence_length*34))

        if args.batch_norm_mlp:
            model.add(BatchNormalization())
        elif args.layer_norm_mlp:
            model.add(LayerNormalization())

        model.add(Activation('relu'))
        
        if args.dropout:
            model.add(Dropout(0.8))

        model.add(Dense(30))

        if args.batch_norm_mlp:
            model.add(BatchNormalization())
        elif args.layer_norm_mlp:
            model.add(LayerNormalization())

        model.add(Activation('relu'))

        if args.dropout:
            model.add(Dropout(0.8))

        model.add(Dense(1, activation='sigmoid'))

    # construct lstm model
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

    elif args.model == '1dcnn':
        model = Sequential()

        # In: [batch, 4, 34]
        model.add(Conv1D(filters=args.conv_filters, kernel_size=args.conv_kernel_size, padding=args.conv_padding, data_format='channels_last', activation=Activation(args.conv_act), input_shape=(sequence_length, 34)))

        if args.dropout:
            model.add(GaussianDropout(0.5))
        # Out: [batch, ?, args.conv_filters]

        if args.conv_layer_num == 2:
            model.add(Conv1D(filters=args.conv_filters, kernel_size=args.conv_kernel_size, padding=args.conv_padding, data_format='channels_last', activation=Activation(args.conv_act)))
            
            if args.dropout:
                model.add(GaussianDropout(0.5))
            # Out: [batch, ?, args.conv_filters]

        # Out: [batch, ?, args.conv_filters]
        model.add(Flatten())
        # Out: [batch, ?*args.conv_filters]

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
