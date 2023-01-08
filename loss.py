import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2, smooth=1e-5, size_average=True, apply_nonlin=None, balance_index=0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

    def call(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        logit = tf.reshape(logit, (-1, num_class))
        target = tf.squeeze(target, 1)
        target = tf.reshape(target, (-1,))

        alpha = self.alpha
        if alpha is None:
            alpha = tf.ones((num_class,))
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = tf.constant(alpha, dtype=tf.float32)
            alpha = alpha / tf.reduce_sum(alpha)
        elif isinstance(alpha, float):
            alpha = tf.ones((num_class,))
            alpha = alpha * (1 - self.alpha)
            alpha = tf.tensor_scatter_nd_update(alpha, [[self.balance_index]], [self.alpha])
        else:
            raise TypeError('Not support alpha type')

        idx = tf.cast(target, tf.int64)

        one_hot_key = tf.one_hot(idx, num_class)
        if self.smooth:
            one_hot_key = tf.maximum(one_hot_key, self.smooth / (num_class - 1))
        pt = tf.reduce_sum(one_hot_key * logit, axis=1) + self.smooth
        print(pt)
        logpt = tf.math.log(pt)

        gamma = self.gamma
        alpha = tf.gather(alpha, idx)
        alpha = tf.squeeze(alpha)
        loss = -1 * alpha * tf.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = tf.reduce_mean(loss)
        return loss