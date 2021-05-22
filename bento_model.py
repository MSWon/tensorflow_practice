import bentoml
import tensorflow as tf

from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.adapters import FileInput

tf.compat.v1.enable_eager_execution() # required


@bentoml.env(pip_packages=['tensorflow', 'numpy', 'pillow'])
@bentoml.artifacts([TensorflowSavedModelArtifact('trackable')])
class MnistTensorflow(bentoml.BentoService):

    @bentoml.api(input=FileInput(), batch=True)
    def predict(self, inputs):
        loaded_func = self.artifacts.trackable.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        inputs = [i.read() for i in inputs]
        inputs = tf.constant(inputs, dtype=tf.string)
        pred = loaded_func(raw_inputs=inputs)
        output = pred['outputs']
        return output