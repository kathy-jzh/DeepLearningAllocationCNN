import tensorflow as tf

# todo put in an object Trainer ?

def loss_func(logits, labels):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        return loss


def evaluate(logits, labels):
    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar("validation_error", (1.0 - accuracy))
        return accuracy


def training(cost,global_step,learning_rate=0.02):
    with tf.name_scope('optimizer'):  # TODO choose the appropriate optimizer

        tf.summary.scalar("cost", cost)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op