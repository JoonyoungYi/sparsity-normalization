import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Layer, Lambda, Dense, Activation, CuDNNLSTM, Dropout, GaussianDropout, BatchNormalization, Conv1D, Flatten
from .LayerNormLSTM import LSTM

class AddBias(Layer):
    """Add bias for SN 1dconv"""
    def __init__(self, **kwargs):
        super(AddBias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias', 
                                      shape=input_shape[-1:],
                                      initializer='zeros',
                                      trainable=True)
        super(AddBias, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

class Zero2One(Layer):
    """Make zero values to one for proper division"""
    def __init__(self, **kwargs):
        super(Zero2One, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Zero2One, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.maximum(x, K.ones(shape=x.shape[1:]))

    def compute_output_shape(self, input_shape):
        return input_shape

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

    elif args.model == '1dcnn_bak':
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

    elif args.model == '1dcnn' and not args.sparsity_normalization:
        inputs = Input(shape=(sequence_length, 34), name='main_input')

        # In: [batch, 4, 34]
        feature1 = Conv1D(filters=args.conv_filters, kernel_size=args.conv_kernel_size, padding=args.conv_padding, data_format='channels_last', activation=Activation(args.conv_act))(inputs)

        if args.dropout:
            feature1 = GaussianDropout(0.5)(feature1)
        # Out: [batch, ?, args.conv_filters]

        if args.conv_layer_num == 2:
            feature2 = Conv1D(filters=args.conv_filters, kernel_size=args.conv_kernel_size, padding=args.conv_padding, data_format='channels_last', activation=Activation(args.conv_act))(feature1)
            
            if args.dropout:
                feature2 = GaussianDropout(0.5)(feature2)
            # Out: [batch, ?, args.conv_filters]

        # Out: [batch, ?, args.conv_filters]
        feature2 = Flatten()(feature2)
        # Out: [batch, ?*args.conv_filters]

        pred = Dense(1, activation='sigmoid')(feature2)

        model = Model(input=[inputs], outputs=[pred])

    elif args.model == '1dcnn' and args.sparsity_normalization:
        # set K_prime
        if args.K_prime_opt == 'n0':
            K_prime = 34 * args.conv_kernel_size
        elif args.K_prime_opt == 'n1':
            raise NotImplementedError()
        elif args.K_prime_opt == 'custom':
            K_prime = args.K_prime
        else:
            assert False, "Invalid K_prime option."

        # get input and apply 1d conv w/o bias
        # In: [batch, 4, 34]
        inputs = Input(shape=(sequence_length, 34), name='main_input')
        feature1 = Conv1D(filters=args.conv_filters, kernel_size=args.conv_kernel_size, padding=args.conv_padding, data_format='channels_last', activation=Activation(args.conv_act),
                  use_bias=False)(inputs)
        # x: [batch, 4, 34]

        # get binarized input and apply 1d conv with filter_value=1 and without bias
        # this calculates zero norm
        # In: [batch, 4, 34]
        binarized_inputs = Input(shape=(sequence_length, 34), name='binarized_input')
        L0_layer = Conv1D(filters=args.conv_filters, kernel_size=args.conv_kernel_size, padding=args.conv_padding, data_format='channels_last', activation=Activation('linear'),
                  kernel_initializer='ones', use_bias=False)
        L0_layer.trainable = False
        divider = L0_layer(binarized_inputs)
        # divider: [batch, 4, 34]

        # TODO: divider conv1d learning X

        # remove zero to divide properly
        divider = Zero2One()(divider)

        feature1 = Lambda(lambda x: x * K_prime)(feature1)
        feature1 = Lambda(lambda x: x[0]/x[1])([feature1, divider])
        feature1 = AddBias()(feature1)

        if args.dropout:
            feature1 = GaussianDropout(0.5)(feature1)
        # Out: [batch, ?, args.conv_filters]

        if args.conv_layer_num == 2:
            feature2 = Conv1D(filters=args.conv_filters, kernel_size=args.conv_kernel_size, padding=args.conv_padding, data_format='channels_last', activation=Activation(args.conv_act))(feature1)
            
            if args.dropout:
                feature2 = GaussianDropout(0.5)(feature2)
            # Out: [batch, ?, args.conv_filters]

        # Out: [batch, ?, args.conv_filters]
        feature2 = Flatten()(feature2)
        # Out: [batch, ?*args.conv_filters]

        pred = Dense(1, activation='sigmoid')(feature2)

        model = Model(input=[inputs, binarized_inputs], outputs=[pred])

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
