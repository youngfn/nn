# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from keras import backend
from keras.optimizers import adam

from train import model_fn


def load_weights(model, weighs_file_path):
    if os.path.exists(weighs_file_path):
        print('load weights from %s' % weighs_file_path)
        model.load_weights(weighs_file_path)
        print('load weights success')
    else:
        print('load weights failed! Please check weighs_file_path')
    return model


def save_pb_model(FLAGS, model):
    if FLAGS.mode == 'train':
        pb_save_dir_obs = FLAGS.train_url
    elif FLAGS.mode == 'save_pb':
        freeze_weights_file_dir = FLAGS.freeze_weights_file_path.rsplit('/', 1)[0]
        pb_save_dir_obs = freeze_weights_file_dir

    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input_img': model.input}, outputs={'output_score': model.output})
    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(pb_save_dir_obs, 'model'))
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=backend.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()
    print('save pb to local path success')

    return pb_save_dir_obs


def load_weights_save_pb(FLAGS):
    optimizer = adam(lr=FLAGS.learning_rate, clipnorm=0.001)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = model_fn(FLAGS, objective, optimizer, metrics)
    model = load_weights(model, FLAGS.freeze_weights_file_path)
    save_pb_model(FLAGS, model)
