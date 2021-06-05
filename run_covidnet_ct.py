"""
Training/testing/inference script for COVIDNet-CT model for COVID-19 detection in CT images.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
import cv2
import json
import numpy as np
from math import ceil
from random import shuffle
import h5py
import tensorflow as tf
from GradCam import GradCAM
import augmentations
import keras
from keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, AveragePooling2D, MaxPooling2D, Dropout,Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.backend import set_session
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report, accuracy_score

from dataset import COVIDxCTDataset
from data_utils import auto_body_crop
from utils import parse_args

# Dict keys
TRAIN_OP_KEY = 'train_op'
TF_SUMMARY_KEY = 'tf_summaries'
LOSS_KEY = 'loss'

# Tensor names
IMAGE_INPUT_TENSOR = 'Placeholder:0'
LABEL_INPUT_TENSOR = 'Placeholder_1:0'
CLASS_PRED_TENSOR = 'ArgMax:0'#predict
CLASS_PROB_TENSOR = 'softmax_tensor:0'#probability
TRAINING_PH_TENSOR = 'is_training:0'
LOSS_TENSOR = 'add:0'

# Names for train checkpoints
CKPT_NAME = 'model.ckpt'
MODEL_NAME = 'COVIDNet-CT'

# Output directory for storing runs
OUTPUT_DIR = 'output'

# Class names ordered by class index
CLASS_NAMES = ('Normal', 'Pneumonia', 'COVID-19')


def dense_grad_filter(gvs):
    """Filter to apply gradient updates to dense layers only"""
    return [(g, v) for g, v in gvs if 'dense' in v.name]


def simple_summary(tag_to_value, tag_prefix=''):
    """Summary object for a dict of python scalars"""
    return tf.Summary(value=[tf.Summary.Value(tag=tag_prefix + tag, simple_value=value)
                             for tag, value in tag_to_value.items() if isinstance(value, (int, float))])

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def create_session():
    """Helper function for session creation"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


class Metrics:
    """Lightweight class for tracking metrics"""
    def __init__(self):
        num_classes = len(CLASS_NAMES)
        self.labels = list(range(num_classes))
        self.class_names = CLASS_NAMES
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)

    def update(self, y_true, y_pred):
        self.confusion_matrix = self.confusion_matrix + confusion_matrix(y_true, y_pred, labels=self.labels)

    def reset(self):
        self.confusion_matrix *= 0

    def values(self):
        conf_matrix = self.confusion_matrix.astype('float')
        metrics = {
            'accuracy': np.diag(conf_matrix).sum() / conf_matrix.sum(),
            'confusion matrix': self.confusion_matrix.copy()
        }
        sensitivity = np.diag(conf_matrix) / np.maximum(conf_matrix.sum(axis=1), 1)
        pos_pred_val = np.diag(conf_matrix) / np.maximum(conf_matrix.sum(axis=0), 1)
        for cls, idx, sens, ppv in zip(self.class_names, self.labels, sensitivity, pos_pred_val):
            metrics['{} {}'.format(cls, 'sensitivity')] = sensitivity[idx]
            metrics['{} {}'.format(cls, 'PPV')] = pos_pred_val[idx]
        return metrics

class LossHistory(keras.callbacks.Callback):
    #函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.count = 0
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
        self.color = {}
        self.color[0]='r'
        self.color[1]='g'
        self.color[2]='k'
        self.color[3]='m'
        self.text = {}
        self.text[0]='CNN-Net-DL'
        self.text[1]='CNN-Net-Dense'
        self.text[2]='CNN-Net-Res'  
 
    #按照batch来进行追加数据
    # def on_batch_end(self, batch, logs={}):
    #     #每一个batch完成后向容器里面追加loss，acc
    #     self.count +=1
    #     if self.count%100==0:
    #         self.losses['batch'].append(logs.get('loss'))
    #         self.accuracy['batch'].append(logs.get('acc'))
    #         self.val_loss['batch'].append(logs.get('val_loss'))
    #         self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    #绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type,num):
        plt.plot(range(len(lists)), lists, self.color[num], label=self.text[num])
        plt.ylabel(label)
        plt.xlabel('epoch')
        plt.title(type+label)
        plt.legend(loc="upper right")
        if label=='loss' and type =='train_batch':
            self.ax1=plt.gca()
        elif label=='acc' and type =='train_batch':
            self.ax2=plt.gca()
        elif label=='loss' and type =='val_batch':
            self.ax3=plt.gca()   
        else:
            self.ax4=plt.gca()    
        plt.savefig('models/'+type+'_'+label+'.jpg')
        
    #由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
    #所以这里的方法会在整个训练结束以后调用
    def end_draw(self,num):
        if num ==0 :
            plt.figure()
        else:
            plt.sca(self.ax1)   
        self.draw_p(self.losses['batch'], 'loss', 'train_batch',num)
        if num ==0 :
            plt.figure()
        else:
            plt.sca(self.ax2)   
        self.draw_p(self.accuracy['batch'], 'acc', 'train_batch',num)
        if num ==0 :
            plt.figure()
        else:
            plt.sca(self.ax3)   
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch',num)
        if num ==0 :
            plt.figure()
        else:
            plt.sca(self.ax4)   
        self.draw_p(self.val_acc['batch'], 'acc', 'val_batch',num)

class COVIDNetCTRunner:
    """Primary training/testing/inference class"""
    def __init__(self, meta_file, ckpt=None, data_dir=None, input_height=512, input_width=512, max_bbox_jitter=0.025,
                 max_rotation=10, max_shear=0.15, max_pixel_shift=10, max_pixel_scale_change=0.2):
        self.meta_file = meta_file
        self.ckpt = ckpt
        self.input_height = input_height
        self.input_width = input_width
        
        if data_dir is None:
            self.dataset = None
        else:
            self.dataset = COVIDxCTDataset(
                data_dir,
                image_height=input_height,
                image_width=input_width,
                max_bbox_jitter=max_bbox_jitter,
                max_rotation=max_rotation,
                max_shear=max_shear,
                max_pixel_shift=max_pixel_shift,
                max_pixel_scale_change=max_pixel_scale_change
            )

    def colab_data(self,is_training,batch_size):
        if is_training:
            temp = np.arange(0, 61)
        else:
            temp = np.arange(0,21)
        while True:
            for i in temp:
                if is_training:
                    filename = 'output/train_' + str(i+1) + '.h5'
                else:
                    filename = 'output/val_' + str(i+1) + '.h5'         
                with h5py.File(filename, 'r') as f:
                    x=f["input_data"][:]
                    y=f["input_labels"][:]
                    m=x.shape[0]
                    n_mini_batches = ceil(m / batch_size) # number of mini batches of size mini_batch_size in your partitionning
                    for k in range(n_mini_batches):
            
                        start_pos = k * batch_size
                        end_pos = min(start_pos + batch_size, m)
                        
                        mini_batch_X = x[start_pos : end_pos, :]
                        mini_batch_Y = y[start_pos : end_pos, :]
                        
                        yield(mini_batch_X, mini_batch_Y)

    def gradcam(self,model, image,seq):
        """Predict"""
        preds = model.predict(image)
        print(preds)
        i = np.argmax(preds[0])
        print(CLASS_NAMES[i],np.max(preds[0]))
        cam = GradCAM(model, int(seq))
        heatmap = cam.compute_heatmap(image)
        heatmap = cv2.resize(heatmap, (self.input_width, self.input_height))
        image=np.uint8(255.0 * image)
        (heatmap, output) = cam.overlay_heatmap(heatmap, image[0], alpha=0.5)
        ## Create a superimposed visualization
        return output

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def load_ckpt(self, sess, saver):
        """Helper for loading weights"""
        # Load weights
        if self.ckpt is not None:
            print('Loading weights from ' + self.ckpt)
            saver.restore(sess, self.ckpt)
    
    def load_graph(self):
        """Creates new graph and session"""
        graph = tf.Graph()
        with graph.as_default():
            # Create session and load model
            sess = create_session()

            # Load meta file
            print('Loading meta graph from ' + self.meta_file)
            saver = tf.train.import_meta_graph(self.meta_file)
        return graph, sess, saver

    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        """
        Implementation of the identity block as defined in Figure 3
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            #first
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third

            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #final step
            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result


    def convolutional_block(self, X_input, kernel_size, in_filter,
                            out_filters, stage, block, training, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #first
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third
            W_conv3 = self.weight_variable([1,1, f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #shortcut path
            W_shortcut = self.weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            #final
            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result

    def deepnn(self, x_input):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:

        Returns:
        """
        x = tf.pad(x_input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
        with tf.variable_scope('reference') :
            training=tf.placeholder(tf.bool, name='training')

            #stage 1
            w_conv1 = self.weight_variable([7, 7, 3, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='VALID')
            print(x.get_shape())

            #stage 2
            x = self.convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

            #stage 3
            x = self.convolutional_block(x, 3, 256, [128,128,512], 3, 'a', training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'b', training=training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'c', training=training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'd', training=training)

            #stage 4
            x = self.convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
            x = self.identity_block (x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

            #stage 5
            x = self.convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)

            x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1,1,1,1], padding='VALID')

            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
            # Dropout - controls the complexity of the model, prevents co-adaptation of
            # features.
            # with tf.name_scope('dropout'):
            #     keep_prob = tf.placeholder(tf.float32)
            #     x = tf.nn.dropout(x, keep_prob)

            logits = tf.layers.dense(x, units=3, activation=tf.nn.softmax)

        return logits,training

    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy_op = tf.reduce_mean(correct_prediction)
        return accuracy_op

    def trainval(self, epochs, output_dir, batch_size=1, learning_rate=0.001, momentum=0.9,
                 fc_only=True, train_split_file='train.txt', val_split_file='val.txt',
                 log_interval=20, val_interval=1000, save_interval=1000):
        """Run training with intermittent validation"""
        ckpt_path = os.path.join(output_dir, CKPT_NAME)
        ## Construct Data
        # if os.path.exists('data/x_train.npy') ==False and os.path.exists('data/y_train.npy') ==False:
        #     self.dataset.train_dataset(train_split_file, batch_size)
        # if os.path.exists('data/x_eval.npy') ==False and os.path.exists('data/y_eval.npy') ==False:
        #     self.dataset.validation_dataset(val_split_file, batch_size)

        ## Construct Model
        plt.switch_backend('agg')
        logs_loss = LossHistory()
        bmodel={}
        bmodel[0] = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (self.input_height, self.input_width, 3)))
        bmodel[1] = InceptionV3(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (self.input_height, self.input_width, 3)))
        bmodel[2] = InceptionResNetV2(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (self.input_height, self.input_width, 3)))
        for i in range(3):
            basemodel=bmodel[i]    
            for layer in basemodel.layers[: -10]:
                layer.trainable = False
            headmodel = basemodel.output
            headmodel = AveragePooling2D(pool_size = (4,4))(headmodel)

            headmodel = Flatten(name= 'flatten')(headmodel)

            headmodel = Dense(256, activation = "relu")(headmodel)
            # headmodel = BatchNormalization()(headmodel)
            headmodel = Dropout(0.4)(headmodel)

            headmodel = Dense(128, activation = "relu")(headmodel)
            # headmodel = BatchNormalization()(headmodel)
            headmodel = Dropout(0.4)(headmodel)

            headmodel = Dense(64, activation = "relu")(headmodel)
            # headmodel = BatchNormalization()(headmodel)
            headmodel = Dropout(0.4)(headmodel)

            headmodel = Dense(3, activation = 'softmax')(headmodel)##4-->3

            model = Model(inputs = basemodel.input, outputs = headmodel)

            model.summary()

            model.compile(loss = 'categorical_crossentropy', 
                        optimizer = optimizers.RMSprop(lr = 1e-4, decay = 1e-6), 
                        metrics = [f1, 'accuracy'])
            print(model.metrics_names)
            #use early stopping to monitor validation loss, stop if val_loss not decreasing after certain number of epochs
            # earlystopping = EarlyStopping(monitor = 'val_acc', mode = 'max', patience = 10)

            #save the model with lowest val_loss
            checkpointer = ModelCheckpoint(filepath = 'weights.hdfs', verbose = 1, save_best_only=True)
            ## Reload Data
            # x = np.load('data/x_train.npy')
            # y = np.load('data/y_train.npy')
            # x_eval= np.load('data/x_eval.npy')
            # y_eval= np.load('data/y_eval.npy')
            # print("Train Size:",x[0].shape)#256,256,3
            # print("Train Num:",len(y))#61792
            # print("Val Num:",len(y_eval))#21036
            # y = tf.keras.utils.to_categorical(y, 3)
            # y_eval = tf.keras.utils.to_categorical(y_eval, 3)
            train_steps=61782// batch_size
            eval_steps=21036// batch_size
            
            hist=model.fit(self.dataset.train_dataset(train_split_file, batch_size),epochs=epochs,validation_data=self.dataset.validation_dataset(val_split_file, batch_size),steps_per_epoch= train_steps,validation_steps=eval_steps, verbose=1,callbacks=[checkpointer,logs_loss])
            # hist=model.fit_generator(generator=self.colab_data(1,batch_size),steps_per_epoch= train_steps,epochs=epochs,verbose=1,callbacks= [checkpointer, earlystopping],validation_data=self.colab_data(0,batch_size),validation_steps=eval_steps)
            # model.save('model.h5')
            ## Draw Result
            
            
            logs_loss.end_draw(i)
            # plt.plot(hist.history['acc'])
            # plt.plot(hist.history['loss'])
            # plt.title('Model Accuracy and Loss Progress During Training')
            # plt.xlabel('Epoch')
            # plt.ylabel('Training Accuracy and Loss')
            # plt.legend(['Training Accuracy', 'Training Loss'])
            # plt.savefig('models/'+str(i)+'train.jpg')
            # plt.clf()
            # plt.plot(hist.history['val_acc'])
            # plt.title('Model Accuracy Progress During Cross-Validation')
            # plt.xlabel('Epoch')
            # plt.ylabel('Validation Accuracy')
            # plt.legend(['Validation Accuracy'])
            # plt.savefig('models/'+str(i)+'valid.jpg')

    def test(self, batch_size=1, test_split_file='test.txt'):
        """Run test on a checkpoint"""
        prediction= []
        original = []
        tf_config = tf.ConfigProto()
        sess = tf.Session(config=tf_config)
        graph = tf.get_default_graph()
        set_session(sess)
        model=load_model('weights.hdfs',custom_objects={"f1":f1})
        model.summary()

        with open(test_split_file, 'r') as f:
            for line in f.readlines():
                fname, cls, xmin, ymin, xmax, ymax = line.strip('\n').split()
                path=os.path.join('data/COVIDx-CT', fname)
                bbox=[int(xmin), int(ymin), int(xmax), int(ymax)]
                # image = tf.image.decode_png(tf.io.read_file(path), channels=1)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                ## augment
                #解决中文显示问题
                # plt.rcParams['font.sans-serif'] = ['SimHei']
                # plt.rcParams['axes.unicode_minus'] = False

                # if autocrop:
                image, _ = auto_body_crop(image)
                image = cv2.resize(image, (self.input_width, self.input_height), cv2.INTER_CUBIC)
                image = image.astype(np.float32) / 255.0
                image = np.expand_dims(np.stack((image, image, image), axis=-1), axis=0)
                predict=model.predict(image)
                predict = np.argmax(predict)
                prediction.append(predict)
                original.append(cls)
                # plt.switch_backend('agg')
                # plt.subplot(231)
                # s = image.numpy()
                # plt.imshow(s[:,:,0])
                # plt.title('裁剪')
                # image, bbox = augmentations.random_rotation(image, 15, bbox)
                # plt.subplot(232)
                # s = image.numpy()
                # plt.imshow(s[:,:,0])
                # plt.title('旋转')
                # image, bbox = augmentations.random_shear(image, 0.2, bbox)
                # plt.subplot(233)
                # s = image.numpy()
                # plt.imshow(s[:,:,0])
                # plt.title('修剪')
                # image = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
                # plt.subplot(234)
                # s = image.numpy()
                # plt.imshow(s[:,:,0])
                # plt.title('放缩')
                # image = augmentations.random_shift_and_scale(image, 15, 0.15)
                # plt.subplot(235)
                # s = image.numpy()
                # plt.imshow(s[:,:,0])
                # plt.title('平移')
                # image = tf.image.random_flip_left_right(image)
                # plt.subplot(236)
                # s = image.numpy()
                # plt.imshow(s[:,:,0])
                # plt.title('水平翻转')
                # print(path)
                # plt.savefig(os.path.join('output', fname))
        score = accuracy_score(original,prediction)
        print("Test Accuracy : {}".format(score))

        # graph, sess, saver = self.load_graph()
        # with graph.as_default():
        #     # Load checkpoint
        #     self.load_ckpt(sess, saver)

        #     # Run test
        #     print('Starting test')
        #     metrics = self._get_validation_fn(sess, batch_size, test_split_file)()
        #     self._log_and_print_metrics(metrics)

        #     if plot_confusion:
        #         # Plot confusion matrix
        #         fig, ax = plt.subplots()
        #         disp = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion matrix'],
        #                                       display_labels=CLASS_NAMES)
        #         disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='horizontal', values_format='.5g')
        #         plt.show()

    def infer(self, image_file, autocrop=False):
        """Run inference on the given image"""
        # Load and preprocess image
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if autocrop:
            image, _ = auto_body_crop(image)
        ## Old version
        image = cv2.resize(image, (512,512), cv2.INTER_CUBIC)    
        # image = cv2.resize(image, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(np.stack((image, image, image), axis=-1), axis=0)
        tf_config = tf.ConfigProto()
        sess = tf.Session(config=tf_config)
        graph = tf.get_default_graph()

        # set_session(sess)
        # model=load_model('/ext/ycy/COVIDNet-CT/weights.hdfs',custom_objects={"f1":f1})
        # # model.summary()
        # predict=model.predict(image)
        # data={}
        # data['Normal']=float(predict[0][0])
        # data['tuber']=float(predict[0][1])
        # data['NCP']=float(predict[0][2])
        # data=json.dumps(data)
        # print(data)


        # Create feed dict
        feed_dict = {IMAGE_INPUT_TENSOR: image, TRAINING_PH_TENSOR: False}

        # Run inference
        graph, sess, saver = self.load_graph()
        with graph.as_default():
            # Load checkpoint
            self.load_ckpt(sess, saver)

            # Run image through model
            class_, probs = sess.run([CLASS_PRED_TENSOR, CLASS_PROB_TENSOR], feed_dict=feed_dict)
            data={}
            data['Normal']=float(probs[0][0])
            data['tuber']=float(probs[0][1])
            data['NCP']=float(probs[0][2])
            data=json.dumps(data)
            print(data)


    def _get_validation_fn(self, sess, batch_size=1, val_split_file='val.txt',fetch_dict=None,train_mode=None,input=None):
        """Creates validation function to call in self.trainval() or self.test()"""
        # Create val dataset
        dataset, num_images = self.dataset.validation_dataset(val_split_file, batch_size)
        dataset = dataset.repeat()  # repeat so there is no need to reconstruct it
        data_next = dataset.make_one_shot_iterator().get_next()
        num_iters = ceil(num_images / batch_size)
        # input=tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 3])
        # Create running accuracy metric
        metrics = Metrics()
        
        # Create feed and fetch dicts
        # fetch_dict = {'classes': CLASS_PRED_TENSOR}
        feed_dict = {train_mode: False}

        def run_validation():
            metrics.reset()
            for i in range(num_iters):
                data = sess.run(data_next)
                feed_dict[input] = data['image']
                results = sess.run(fetch_dict, feed_dict)
                metrics.update(data['label'], results['classes'])
            return metrics.values()

        return run_validation

    @staticmethod
    def _log_and_print_metrics(metrics, step=None, summary_writer=None, tag_prefix='val/'):
        """Helper for logging and printing"""
        # Pop temporarily and print
        cm = metrics.pop('confusion matrix')
        print('\tconfusion matrix:')
        print('\t' + str(cm).replace('\n', '\n\t'))

        # Print scalar metrics
        for name, val in sorted(metrics.items()):
            print('\t{}: {}'.format(name, val))

        # Log scalar metrics
        if summary_writer is not None:
            summary = simple_summary(metrics, tag_prefix)
            summary_writer.add_summary(summary, step)

        # Restore confusion matrix
        metrics['confusion matrix'] = cm

    @staticmethod
    def _get_train_summary_op(loss, tag_prefix='train/'):
        loss_summary = tf.summary.scalar(tag_prefix + 'loss', loss)
        return loss_summary


if __name__ == '__main__':
    # Suppress most console output
    mode, args = parse_args(sys.argv[1:])


    # Create full paths
    meta_file = os.path.join(args.model_dir, args.meta_name)
    ckpt = os.path.join(args.model_dir, args.ckpt_name)
    # Create runner
    if mode == 'train':
        augmentation_kwargs = dict(
            max_bbox_jitter=args.max_bbox_jitter,
            max_rotation=args.max_rotation,
            max_shear=args.max_shear,
            max_pixel_shift=args.max_pixel_shift,
            max_pixel_scale_change=args.max_pixel_scale_change
        )
    else:
        augmentation_kwargs = {}
    runner = COVIDNetCTRunner(
        meta_file,
        ckpt=ckpt,
        data_dir=args.data_dir,
        input_height=args.input_height,
        input_width=args.input_width,
        **augmentation_kwargs
    )

    if mode == 'train':
        # tf.disable_eager_execution()
        # Create output_dir and save run settings
        output_dir = os.path.join(OUTPUT_DIR, MODEL_NAME + args.output_suffix)
        # os.makedirs(output_dir, exist_ok=False)
        # with open(os.path.join(output_dir, 'run_settings.json'), 'w') as f:
        #     json.dump(vars(args), f)

        # Run trainval
        runner.trainval(
            args.epochs,
            output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            fc_only=args.fc_only,
            train_split_file=args.train_split_file,
            val_split_file=args.val_split_file,
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            save_interval=args.save_interval
        )
    elif mode == 'test':
        # tf.enable_eager_execution()
        # Run validation
        runner.test(
            batch_size=args.batch_size,
            test_split_file=args.test_split_file,
        )
    else:
        os.environ['KMP_WARNINGS'] = '0'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        # Run inference
        runner.infer(args.image_file, args.auto_crop)
        # runner.infer(path, args.auto_crop)
