import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


N_NODES_HIDDEN_LAYER_1 = N_NODES_HIDDEN_LAYER_2 = N_NODES_HIDDEN_LAYER_3 = 500
N_CLASSES = 10
BATCH_SIZE = 100
N_PIXELS = 784


def get_layer(shape_w, shape_b):
    return {'weights': tf.Variable(tf.random_normal(shape_w)),
            'biases': tf.Variable(tf.random_normal(shape_b))}


def neural_network_model(data):
    hidden_layer_1 = get_layer([N_PIXELS, N_NODES_HIDDEN_LAYER_1], [N_NODES_HIDDEN_LAYER_1])
    hidden_layer_2 = get_layer([N_NODES_HIDDEN_LAYER_1, N_NODES_HIDDEN_LAYER_2], [N_NODES_HIDDEN_LAYER_2])
    hidden_layer_3 = get_layer([N_NODES_HIDDEN_LAYER_2, N_NODES_HIDDEN_LAYER_3], [N_NODES_HIDDEN_LAYER_3])
    output_layer = get_layer([N_NODES_HIDDEN_LAYER_3, N_CLASSES], [N_CLASSES])

    # model -> (input_data * weights) + biases

    layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)  # activation function

    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']

    return output  # one hot array


def train_neural_network(data):
    prediction = neural_network_model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    total_epochs = 10

    with tf.Session() as sess:
        # Beginning of train
        sess.run(tf.global_variables_initializer())

        for epoch in range(total_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _, epoch_cost = sess.run([optimizer, cost], feed_dict={data: epoch_x, y: epoch_y})
                epoch_loss += epoch_cost

            print('{}/{} epochs completed! Loss: {}'.format(epoch + 1, total_epochs, epoch_loss))
            # End of train

        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy: ', accuracy.eval({data: mnist.test.images, y: mnist.test.labels}))


mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

X = tf.placeholder('float', [None, 784])  # images in the data set are 28x28
y = tf.placeholder('float')

train_neural_network(X)
