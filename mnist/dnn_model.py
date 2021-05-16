import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

BATCH_SIZE = 100
EPOCHS = 20
TOTAL_BATCH = int(mnist.train.num_examples / BATCH_SIZE)
MODEL_NAME = 'mnist.dnn'


def build_graph(input_x):
    W1 = tf.Variable(tf.random_normal([784, 10], stddev=0.01), name='W1')
    B1 = tf.Variable(tf.random_normal([10]), name='B1')
    logits = tf.matmul(input_x, W1) + B1
    return logits


def build_opt(logits, labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    opt = tf.train.AdamOptimizer(0.05).minimize(loss)
    return loss, opt


def get_accuracy(logits, labels):
    is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    return accuracy


# input placeholder
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

logits = build_graph(X)
scores = tf.nn.softmax(logits, axis=-1)
predictions = tf.argmax(logits, axis=-1)
loss, opt = build_opt(logits, Y)
accuracy = get_accuracy(logits, Y)


def main():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(EPOCHS):
            total_cost = 0
            for i in range(TOTAL_BATCH):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                _, cost_val = sess.run([opt, loss], feed_dict={X: batch_xs, Y: batch_ys})
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / TOTAL_BATCH), 'Train Acc. =', sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels}))

        print('Training Done!')
        print('Test Acc. = ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
        save_path = saver.save(sess, "./model/" + MODEL_NAME)
        print(f'saved path : {save_path}')
        sess.close()


if __name__ == "__main__":
    main()
