import tensorflow as tf


class DropConnect:
    def __init__(self,
                 drop_rate=0.5,
                 activation_fn=tf.nn.relu,
                 Z=1000,
                 mode='default'):
        assert 0 <= drop_rate <= 1
        assert mode in ['default', 'like-dropout', 'sparsity-normalization']

        self.drop_rate = drop_rate
        self.activation_fn = activation_fn
        self.Z = Z
        self.mode = mode

    def __call__(self, layer, weight, bias, training):
        drop_rate = self.drop_rate
        activation_fn = self.activation_fn
        Z = self.Z
        mode = self.mode

        if drop_rate < 1e-8:
            return activation_fn(tf.matmul(layer, weight) + bias)

        if mode == 'default' or mode == 'like-dropout':
            weight_r = tf.random.uniform(
                tf.shape(weight), minval=0., maxval=1.)
            weight_m = tf.where(weight_r < drop_rate,
                                tf.zeros_like(weight), tf.ones_like(weight))
            train_weight = tf.multiply(weight, weight_m)
            # bias_r = tf.random.uniform(tf.shape(bias), minval=0., maxval=1.)
            # bias_m = tf.where(bias_r < drop_rate,
            #                   tf.zeros_like(bias), tf.ones_like(bias))
            # train_bias = tf.multiply(bias, bias_m)
            # 헷갈려서 임시로 주석 처리.
            train_bias = bias
            train_layer = activation_fn(
                tf.matmul(layer, train_weight) + train_bias)
        elif mode == 'sparsity-normalization':

            weight_r = tf.random.uniform(
                tf.shape(weight), minval=0., maxval=1.)
            weight_m = tf.where(weight_r < drop_rate,
                                tf.zeros_like(weight), tf.ones_like(weight))
            train_weight = tf.multiply(weight, weight_m)
            train_layer = tf.matmul(layer, train_weight)
            weight_m_number = tf.reduce_sum(weight_m, axis=0, keepdims=True)
            # self.weight_m_number = weight_m_number
            weight_m_number = tf.maximum(weight_m_number,
                                         tf.ones_like(weight_m_number))

            train_bias = bias

            train_layer = train_layer * (
                tf.cast(tf.shape(weight_m)[0], tf.float32) / weight_m_number)
            train_layer = activation_fn(train_layer + train_bias)
        else:
            raise NotImplementedError()

        if mode == 'default':
            test_mu = drop_rate * tf.matmul(layer, weight) + bias
            test_sigma = tf.sqrt(drop_rate * (1 - drop_rate) * tf.matmul(
                tf.square(layer), tf.square(weight)))
            test_mu = tf.expand_dims(test_mu, -1)
            test_sigma = tf.expand_dims(test_sigma, -1)
            test_random_seed = tf.random.truncated_normal(
                [tf.shape(test_mu)[0],
                 tf.shape(test_mu)[1], Z])
            test_output = test_mu + test_random_seed * test_sigma
            test_activation = activation_fn(test_output)
            test_layer = tf.reduce_mean(test_activation, axis=2)
        elif mode == 'like-dropout':
            test_layer = tf.matmul(layer, drop_rate * weight) + bias
        elif mode == 'sparsity-normalization':
            test_layer = tf.matmul(layer, weight) + bias
        else:
            raise NotImplementedError()

        return tf.cond(training, lambda: train_layer, lambda: test_layer)


# def dropconnect(layer, weight, training, args):
#     if args.drop_rate > 1e-8:
#         test_layer = tf.matmul(layer, weight)
#
#         if args.dropconnect_inference == 'simple':
#             train_layer = train_layer / (1. - args.dropconnect_rate)
#         elif args.dropconnect_inference == 'sn':
#             m_number = tf.reduce_sum(m, axis=0, keepdims=True)
#             m_number = tf.maximum(m_number, tf.ones_like(m_number))
#             train_layer = train_layer * (
#                 tf.cast(tf.shape(m)[0], tf.float32) / m_number)
#         else:
#             raise NotImplementedError()
#
#         return tf.cond(training, lambda: train_layer, lambda: test_layer)
#     else:
#         return tf.matmul(layer, weight)

# def dropconnect(layer, weight, training, args):
#     if args.dropconnect_rate > 1e-8:
#         test_layer = tf.matmul(layer, weight)
#
#         r = tf.random.uniform(tf.shape(weight), minval=0., maxval=1.)
#         m = tf.where(r < args.dropconnect_rate,
#                      tf.zeros_like(weight), tf.ones_like(weight))
#         train_weight = tf.multiply(weight, m)
#         train_layer = tf.matmul(layer, train_weight)
#
#         if args.dropconnect_inference == 'simple':
#             train_layer = train_layer / (1. - args.dropconnect_rate)
#         elif args.dropconnect_inference == 'sn':
#         else:
#             raise NotImplementedError()
#
#         return tf.cond(training, lambda: train_layer, lambda: test_layer)
#     else:
#         return tf.matmul(layer, weight)
