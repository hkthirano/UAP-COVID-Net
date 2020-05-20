import os

import cv2
import numpy as np
import tensorflow as tf
from art.classifiers import TFClassifier
from keras.utils import np_utils
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}


def load_data(datapath, trainfile, testfile):
    if os.path.exists(datapath + '/x_train.npy'):
        x_train = np.load(datapath + '/x_train.npy')
        y_train = np.load(datapath + '/y_train.npy')
        x_test = np.load(datapath + '/x_test.npy')
        y_test = np.load(datapath + '/y_test.npy')
    else:
        files = {'train': trainfile, 'test': testfile}
        dataset = {}
        for data_type in ['train', 'test']:
            print('make {} dataset'.format(data_type))
            x_list = []
            y_list = []
            datafile = open(files[data_type], 'r')
            datafile = datafile.readlines()
            for i in tqdm(range(len(datafile))):
                line = datafile[i].split()
                x = cv2.imread(os.path.join(datapath, data_type, line[1]))
                h, w, c = x.shape
                x = x[int(h / 6):, :]
                x = cv2.resize(x, (224, 224))
                x = x.astype('float32') / 255.0
                x_list.append(x)
                y_list.append(mapping[line[2]])
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            y_list = np_utils.to_categorical(y_list)
            np.save('{}/x_{}'.format(datapath, data_type), x_list)
            np.save('{}/y_{}'.format(datapath, data_type), y_list)
            dataset['x_{}'.format(data_type)] = x_list
            dataset['y_{}'.format(data_type)] = y_list
        x_train = dataset['x_train']
        y_train = dataset['y_train']
        x_test = dataset['x_test']
        y_test = dataset['y_test']
    mean_l2_train = 0
    mean_inf_train = 0
    for im in x_train:
        mean_l2_train += np.linalg.norm(im[:, :, 0].flatten(), ord=2)
        mean_inf_train += np.abs(im[:, :, 0].flatten()).max()
    mean_l2_train /= len(x_train)
    mean_inf_train /= len(x_train)
    return (x_train, y_train), (x_test, y_test), (mean_l2_train, mean_inf_train)


def create_model(weightspath, metaname, ckptname):
    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(
        os.path.join(weightspath, metaname))
    saver.restore(sess, os.path.join(weightspath, ckptname))
    graph = tf.get_default_graph()
    return sess, graph


def set_up(args):
    if not os.path.isdir('output'):
        os.makedirs('output')
    # # Load the chestx dataset
    (x_train, y_train), (x_test, y_test), (mean_l2_train,
                                           mean_inf_train) = load_data(args.datapath, args.trainfile, args.testfile)
    if args.norm == '2':
        norm = 2
        eps = mean_l2_train * args.eps
    elif args.norm == 'inf':
        norm = np.inf
        eps = mean_inf_train * args.eps
    # # Create the model
    sess, graph = create_model(args.weightspath, args.metaname, args.ckptname)
    # # Create the ART classifier
    input_tensor = graph.get_tensor_by_name("input_1:0")
    logit_tensor = graph.get_tensor_by_name("dense_3/MatMul:0")
    output_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")
    label_tensor = graph.get_tensor_by_name("dense_3_target:0")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logit_tensor, labels=label_tensor))
    classifier = TFClassifier(input_ph=input_tensor, output=output_tensor,
                              labels_ph=label_tensor, loss=loss, sess=sess)
    return (x_train, y_train), (x_test, y_test), (mean_l2_train, mean_inf_train), norm, eps, classifier


def get_preds(classifier, x, y):
    preds = np.argmax(classifier.predict(x), axis=1)
    acc = np.sum(preds == np.argmax(y, axis=1)) / len(y)
    return acc, preds


def get_foolingrate_rate(preds_adv, y):
    fooling_adv = np.sum(preds_adv != np.argmax(y, axis=1)) / len(y)
    return fooling_adv


def get_target_success_rate(preds_adv, target):
    target_success_rate = np.sum(preds_adv == target) / len(preds_adv)
    return target_success_rate


def show_confusion_matrix(preds, preds_adv):
    matrix = confusion_matrix(preds, preds_adv)
    matrix = matrix.astype('float')
    # cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix.astype('int'))
    # class_acc = np.array(cm_norm.diagonal())
    class_acc = [matrix[i, i] / np.sum(matrix[i, :])
                 if np.sum(matrix[i, :]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i, i] / np.sum(matrix[:, i]) if np.sum(matrix[:, i])
            else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))


def make_adv_img(clean_img, noise, adv_img, filename):
    # clean
    im_clean = clean_img * 255.0
    im_clean = np.squeeze(np.clip(im_clean, 0, 255).astype(np.uint8))
    # noise
    im_noise = (noise - noise.min()) / \
        (noise.max() - noise.min()) * 255
    im_noise = np.squeeze(im_noise.astype(np.uint8))
    # adv
    im_adv = adv_img * 255.0
    im_adv = np.squeeze(np.clip(im_adv, 0, 255).astype(np.uint8))
    # all
    img_all = np.concatenate((im_clean, im_noise, im_adv), axis=1)
    img_all = Image.fromarray(np.uint8(img_all))
    img_all.save(filename)
