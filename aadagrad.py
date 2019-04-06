from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
import tensorflow as tf

class AAdagrad(Optimizer):
    """AAdagrad optimizer.
    """

    def __init__(self, lr=0.01, epsilon=None, decay=0., **kwargs):
        super(AAdagrad, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        past_grads = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, past_g in zip(params, grads, accumulators, past_grads):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))
            new_p = tf.where(past_g*g>0,p - lr * (g + past_g) / (K.sqrt(new_a) + self.epsilon),p - lr * g / (K.sqrt(new_a) + self.epsilon))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            new_g = g
            self.updates.append(K.update(past_g, new_g))
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(AAdagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
