import tensorflow as tf


class DropConnect:
    def __init__(self, drop_rate=0.5, activation_fn=tf.nn.relu, Z=10):
        assert 0 <= drop_rate <= 1
        self.drop_rate = drop_rate
        self.activation_fn = activation_fn
        self.Z = Z

    def __call__(self, layer, weight, bias, training):
        drop_rate = self.drop_rate
        activation_fn = self.activation_fn

        if drop_rate < 1e-8:
            return activation_fn(tf.matmul(layer, weight) + bias)

        weight_r = tf.random.uniform(tf.shape(weight), minval=0., maxval=1.)
        weight_m = tf.where(weight_r < drop_rate,
                            tf.zeros_like(weight), tf.ones_like(weight))
        train_weight = tf.multiply(weight, weight_m)
        bias_r = tf.random.uniform(tf.shape(bias), minval=0., maxval=1.)
        bias_m = tf.where(bias_r < drop_rate,
                          tf.zeros_like(bias), tf.ones_like(bias))
        train_bias = tf.multiply(bias, bias_m)
        train_layer = activation_fn(
            tf.matmul(layer, train_weight) + train_bias)

        test_layer = train_layer
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
