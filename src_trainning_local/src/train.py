# -*- coding: utf-8 -*-
import os
import multiprocessing
from glob import glob
import numpy as np
from keras import backend
from keras.models import Model
from keras.optimizers import adam, SGD
from keras.layers import Flatten, Dense
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from keras.preprocessing import image
##from keras.losses import categorical_crossentropy, binary_crossentropy

from data_gen import data_flow
from models.resnet50 import ResNet50

backend.set_image_data_format('channels_last')


def model_fn(FLAGS, objective, optimizer, metrics):
    """
    pre-trained resnet50 model
    """
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


class LossHistory(Callback):
    def __init__(self, FLAGS):
        super(LossHistory, self).__init__()
        self.FLAGS = FLAGS

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.FLAGS.snapshot_freq == 0:
            save_path = os.path.join(self.FLAGS.train_url, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc')))
            self.model.save_weights(save_path)

        if self.FLAGS.keep_weights_file_num > -1:
            weights_files = glob(os.path.join(self.FLAGS.train_url, '*.h5'))
            if len(weights_files) >= self.FLAGS.keep_weights_file_num:
                weights_files.sort(key=lambda file_name: os.stat(file_name).st_ctime, reverse=True)
                for file_path in weights_files[self.FLAGS.keep_weights_file_num:]:
                    os.remove(file_path)  # only remove weights files on local path

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -backend.mean(alpha * backend.pow(1. - pt_1, gamma) * backend.log(pt_1)) - backend.mean((1 - alpha) * backend.pow(pt_0, gamma) * backend.log(1 - pt_0))
    return focal_loss_fixed

#initial_learning_rate = 0.0001
#lr_schedule = schedules.ExponentialDecay(
#    initial_learning_rate,
#    decay_steps=5000,
#    decay_rate=0.98,
#    staircase=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

def train_model(FLAGS):
    # data flow generator
    # train_sequence, validation_sequence = data_flow(FLAGS.data_url, FLAGS.batch_size,
                                                    # FLAGS.num_classes, FLAGS.input_size)
    train_datagen = image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip = False,
    fill_mode='nearest'
    )
    
    input_size = FLAGS.input_size
    train_generator = train_datagen.flow_from_directory(
    'D:\\01_Learning\\lyn\\00_project\\nn-master\\train_data_new2',
    target_size=(input_size, input_size),
    batch_size=FLAGS.batch_size,
    class_mode='categorical',
    # save_to_dir='D:\\01_Learning\\lyn\\00_project\\nn-master\\train_data_train\\',
    shuffle=True,
    # seed=1
    )
    test_datagen = image.ImageDataGenerator() # 验证集不用增强
    validation_generator = test_datagen.flow_from_directory(
        'D:\\01_Learning\\lyn\\00_project\\nn-master\\train_data_new2',
        target_size=(input_size, input_size),
        batch_size=20,
        shuffle=True,
        class_mode='categorical'
        # save_to_dir='D:\\01_Learning\\lyn\\00_project\\nn-master\\train_data_test\\',
    )
    
    
    optimizer = adam(lr=FLAGS.learning_rate, clipnorm=0.001)
    #optimizer = SGD(learing_rate=lr_schedule)
    objective = 'categorical_crossentropy'
    #objective = focal_loss(gamma=2, alpha=.25)
    metrics = ['accuracy']
    model = model_fn(FLAGS, objective, optimizer, metrics)
    if FLAGS.restore_model_path != '' and os.path.exists(FLAGS.restore_model_path):

        model.load_weights(FLAGS.restore_model_path)
        print('restore parameters from %s success' % FLAGS.restore_model_path)

    if not os.path.exists(FLAGS.train_url):
        os.makedirs(FLAGS.train_url)
    tensorboard = TensorBoard(log_dir=FLAGS.train_url, batch_size=FLAGS.batch_size)
    # history = LossHistory(FLAGS)
    filepath = FLAGS.train_url + 'weights-{epoch:03d}-{val_accuracy:.04f}.h5'
    check_pointer = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
    # model.fit_generator(
        # train_sequence,
        # steps_per_epoch=len(train_sequence),
        # epochs=FLAGS.max_epochs,
        # verbose=1,
        # callbacks=[reduce_lr, tensorboard,check_pointer],
        # validation_data=validation_sequence,
        # max_queue_size=10,
        # workers=int(multiprocessing.cpu_count() * 0.7),
        # use_multiprocessing=False,
        # shuffle=True
    # )
    model.fit_generator(
    train_generator,
    steps_per_epoch=26,
    epochs=FLAGS.max_epochs,
    validation_data=validation_generator,
    validation_steps=26,
    shuffle=True,
    callbacks=[reduce_lr, tensorboard,check_pointer]
    )

    print('training done!')

    # 将训练日志拷贝到OBS，然后可以用 ModelArts 训练作业自带的tensorboard查看训练情况
    if FLAGS.train_url.startswith('s3://'):
        files = mox.file.list_directory(FLAGS.train_local)
        for file_name in files:
            if file_name.startswith('enevts'):
                mox.file.copy(os.path.join(FLAGS.train_local, file_name), os.path.join(FLAGS.train_url, file_name))
        print('save events log file to OBS path: ', FLAGS.train_url)

    pb_save_dir_local = ''
    if FLAGS.deploy_script_path != '':
        from save_model import save_pb_model
        # 默认将最新的模型保存为pb模型，您可以使用python run.py --mode=save_pb ... 将指定的h5模型转为pb模型
        pb_save_dir_local = save_pb_model(FLAGS, model)

    if FLAGS.deploy_script_path != '' and FLAGS.test_data_url != '':
        print('test dataset predicting...')
        from inference import infer_on_dataset
        accuracy, result_file_path = infer_on_dataset(FLAGS.test_data_local, FLAGS.test_data_local, os.path.join(pb_save_dir_local, 'model'))
        if accuracy is not None:
            metric_file_name = os.path.join(FLAGS.train_url, 'metric.json')
            metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % accuracy
            with mox.file.File(metric_file_name, "w") as f:
                f.write(metric_file_content + '\n')
            if FLAGS.train_url.startswith('s3://'):
                result_file_path_obs = os.path.join(FLAGS.train_url, 'model', os.path.basename(result_file_path))
                mox.file.copy(result_file_path, result_file_path_obs)
                print('accuracy result file has been copied to %s' % result_file_path_obs)
        else:
            print('accuracy is None')
    print('end')
