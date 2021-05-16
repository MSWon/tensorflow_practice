import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import signature_constants

class ExportModel(object):
    def __init__(self):
        # set Graph
        self.graph = tf.Graph()
        # initialize session
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with self.sess.as_default() as sess:
                metagraph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 'exported.model')
                self._mapping = dict()
                self._mapping.update(dict(metagraph.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs))
                self._mapping.update(dict(metagraph.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs))

                self.inputs = (
                    self.sign2tensor("export/inputs"),
                )

                self.outputs = (
                    self.sign2tensor("export/scores"),
                    self.sign2tensor("export/outputs")
                )

    def sign2tensor(self, sign_name):
        tensor_name = self._mapping[sign_name].name
        print(f"mapping : {sign_name} -> {tensor_name}")
        return self.graph.get_tensor_by_name(tensor_name)

    def infer(self, input_image):
        feed_dict_candidate = {
            self.inputs[0]: input_image
        } 
        score, output = self.sess.run(self.outputs, feed_dict=feed_dict_candidate)
        return score, output

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    random_idx = random.randint(0, len(mnist.test.images))
    sample_image = mnist.test.images[random_idx]
    sample_label = mnist.test.labels[random_idx]

    exp_model = ExportModel()
    score, output = exp_model.infer(sample_image[None, :])

    print(f"Gold label: {sample_label}")
    print(f"Model predicted label: {output[0]}")
    print(f"Model softmax score: {score}")