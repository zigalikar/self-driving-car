import cv2

import os.path as path
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import other.util as util
from modules.module_base import ModuleBase
from other.util import construct_weights_path
from modules.traffic_sign_classifier.traffic_sign_classifier_util import TrafficSignClassifierUtil as tfcu

class TrafficSignClassifier(ModuleBase):

    def __init__(self, module_name, loader, config):
        super().__init__(module_name, loader, config)
        
        self.placeholders = self.init_pipeline()
        
        weights_path = util.construct_weights_path(module_name, config, return_root=True)
        ckpt = path.join(weights_path, 'checkpoint')
        
        self.output_module_init(pretrained=path.isfile(ckpt))
        if path.isfile(ckpt):
            test_x, test_y = self.preprocess(self.dataset_test['features']), self.dataset_test['labels']
            
            self.weights_path = ckpt
            self.test((test_x, test_y))
        else:
            self.init_training()            

    # Initializes the training process
    def init_training(self):        
        x_norm, x_val, y_norm, y_val = self.get_train_test_split(self.dataset_train['features'], self.dataset_train['labels'])
        self.train((x_norm, y_norm), (x_val, y_val))
    
    # Trains the model on the training set
    def train(self, train: tuple, val: tuple):
        epochs = self.get_property('epochs')
        batches_per_epoch = self.get_property('batches_per_epoch')
        batch_size = self.get_property('batch_size')
        x_train, y_train = train[0], train[1]
        x_val, y_val = val[0], val[1]
        x, y, keep_prob = self.placeholders['x'], self.placeholders['y'], self.placeholders['keep_prob']

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            log_path = path.join(construct_weights_path(self.module_name, self.config, return_root=True), self.config.epochs_log)

            with open(log_path, 'w') as f:
                for epoch in range(epochs):
                    self.module_log("\033[94m---- Starting epoch {0}/{1} ----\033[0m".format(epoch + 1, epochs))
                    batch_counter = 0

                    for batch_x, batch_y in self.image_datagen.flow(x_train, y_train, batch_size=batch_size):
                        batch_counter += 1
                        sess.run(self.training_pipeline['train_step'], feed_dict={ x: batch_x, y: batch_y, keep_prob: self.get_property('keep_prob_train') })
                        self.module_log('Processing batch {0}/{1} - {2:.2f}% complete             '.format(batch_counter, batches_per_epoch, batch_counter / batches_per_epoch * 100), end='\r', log_to_file=False)

                        if batch_counter == batches_per_epoch:
                            break

                    util.log()
                    train_accuracy = self.evaluate(x_train, y_train, eval_type='training')
                    val_accuracy = self.evaluate(x_val, y_val, eval_type='validation')
                    result = 'Epoch {} -- Train accuracy: {:.3f} | Validation accuracy: {:.3f}'.format(epoch + 1, train_accuracy, val_accuracy)
                    self.module_log(result)
                    f.write(result + '\n')
                    
                    weights_path = construct_weights_path(self.module_name, self.config)
                    self.saver.save(sess, save_path=weights_path, global_step=epoch)
    
    # Tests the model on the test set
    def test(self, test: tuple):
        x_test, y_test = test[0], test[1]

        if self.weights_path != None:
            with tf.Session() as sess:
                path = self.get_latest_weights()

                if self.saver is not None and path != None:
                    self.saver.restore(sess, path)
                    
                    acc = self.evaluate(x_test, y_test, 'testing')
                    self.module_log('Accuracy on testing dataset: {:.3f}%'.format(acc * 100))

    # Returns the latest weights file path, TODO: get best weights file path
    def get_latest_weights(self):
        root_path = construct_weights_path(self.module_name, self.config, return_root=True)

        if self.weights_path != None:
            with open(self.weights_path, 'r') as f:
                epochs = f.read().split('\n')
                return path.join(root_path, epochs[-2].split('"')[1])

        return None

    # Preprocesses the data
    def preprocess(self, x):
        x = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in x])
        x = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in x])
        x = np.float32(x)
        x -= np.mean(x, axis=0)
        x /= (np.std(x, axis=0) + np.finfo('float32').eps)

        return x

    # Gets the normalized training/testing data and their labels
    def get_train_test_split(self, x, y):
        x = self.preprocess(x)
        x_norm, x_val, y_norm, y_val = train_test_split(x, y, test_size=self.get_property('val_ratio'), random_state=0)
        return x_norm, x_val, y_norm, y_val

    # Inits the pipeline
    def init_pipeline(self):
        x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
        y = tf.placeholder(dtype=tf.int32, shape=None)
        keep_prob = tf.placeholder(tf.float32)
        lr = self.get_property('lr')

        logits = self.get_logits(x, keep_prob)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss_function = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.minimize(loss=loss_function)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.image_datagen = ImageDataGenerator(
            rotation_range=15.,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1)
            
        self.saver = tf.train.Saver()
        self.training_pipeline = { 'train_step': train_step, 'accuracy_operation': accuracy_operation }
        self.placeholders = { 'x': x, 'y': y, 'keep_prob': keep_prob }
        return self.placeholders
    
    # Evaluates the inputs with the correct outputs
    def evaluate(self, data_x, data_y, eval_type='training'):
        x, y, keep_prob = self.placeholders['x'], self.placeholders['y'], self.placeholders['keep_prob']
        num_examples = data_x.shape[0]
        total_accuracy = 0
        batch_size = self.get_property('batch_size')

        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            self.module_log('Evaluating epoch - {0}: {1}/{2} - {3:.2f}% complete                '.format(eval_type, offset, num_examples, offset / num_examples * 100), end='\r', log_to_file=False)
            batch_x, batch_y = data_x[offset:offset+batch_size], data_y[offset:offset+batch_size]
            accuracy = sess.run(self.training_pipeline['accuracy_operation'], feed_dict={ x: batch_x, y: batch_y, keep_prob: self.get_property('keep_prob_eval') })
            total_accuracy += accuracy * len(batch_x)
        
        util.log()
        return total_accuracy / num_examples
    
    # Runs the input through the neural network and returns the logits
    def get_logits(self, x, keep_prob):
        c1_out = 64

        conv1_W = tfcu.get_weights(shape=(3, 3, 1, c1_out))
        conv1_b = tfcu.get_bias(shape=(c1_out,))
        conv1 = tf.nn.relu(tfcu.conv2d(x, conv1_W) + conv1_b)

        pool1 = tfcu.max_pool_2x2(conv1)

        drop1 = tf.nn.dropout(pool1, keep_prob=keep_prob)

        c2_out = 128
        conv2_W = tfcu.get_weights(shape=(3, 3, c1_out, c2_out))
        conv2_b = tfcu.get_bias(shape=(c2_out,))
        conv2 = tf.nn.relu(tfcu.conv2d(drop1, conv2_W) + conv2_b)

        pool2 = tfcu.max_pool_2x2(conv2)

        drop2 = tf.nn.dropout(pool2, keep_prob=keep_prob)

        fc0 = tf.concat([flatten(drop1), flatten(drop2)], 1)

        fc1_out = 64
        fc1_W = tfcu.get_weights(shape=(fc0._shape[1].value, fc1_out))
        fc1_b = tfcu.get_bias(shape=(fc1_out,))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        drop_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

        fc2_out = tfcu.get_num_classes(self.dataset_train)
        fc2_W = tfcu.get_weights(shape=(drop_fc1._shape[1].value, fc2_out))
        fc2_b = tfcu.get_bias(shape=(fc2_out,))
        logits = tf.matmul(drop_fc1, fc2_W) + fc2_b

        return logits