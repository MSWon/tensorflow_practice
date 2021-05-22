import tensorflow as tf
import argparse
from tensorflow.python.saved_model import signature_constants
from mnist.dnn_model import raw_X, X, scores, predictions

"""
Usage:
python export_model.py \
    --input-checkpoint ${CHECK_POINT_PATH} \
    --saved-model-path ${EXPORT_MODEL_PATH}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--input-checkpoint", required=True)
parser.add_argument("--saved-model-path", required=True)
args = parser.parse_args()

'''for using 'infer_export_model.py'
inputs_candidate = {
    "inputs": X
}
'''
inputs_candidate = {
    "raw_inputs": raw_X
}
outputs_candidate = {
    "scores": scores,
    "outputs": predictions
}

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

print("START Exporting model")

with tf.Session() as sess:
    # Need to build tensor graph first, in order to restore variables
    saver = tf.train.Saver(var_list)
    saver.restore(sess, args.input_checkpoint)
    # SavedModel builder
    builder = tf.saved_model.builder.SavedModelBuilder(args.saved_model_path)
    
    builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.saved_model.signature_def_utils.predict_signature_def(
                        inputs=inputs_candidate,
                        outputs=outputs_candidate
                    )
            }
        )
    
    builder.save()

print("END Exporting model")