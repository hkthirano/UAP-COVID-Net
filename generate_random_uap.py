import argparse

import numpy as np
import tensorflow as tf
from art.utils import random_sphere
from art.classifiers import TFClassifier

from uap_utils import (get_foolingrate_rate, get_preds, get_target_success_rate,
                       show_confusion_matrix, load_data, create_model, make_adv_img)

parser = argparse.ArgumentParser(description='COVID-Net Evaluation')
parser.add_argument('--weightspath', default='../COVID-Net/models/COVIDNet-CXR-Small',
                    type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta',
                    type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-1697',
                    type=str, help='Name of model ckpts')
parser.add_argument('--datapath', default='../COVID-Net/data',
                    type=str, help='Path to data folder')
parser.add_argument('--trainfile', default='../COVID-Net/train_split_v3.txt',
                    type=str, help='Path to testfile')
parser.add_argument('--testfile', default='../COVID-Net/test_split_v3.txt',
                    type=str, help='Path to testfile')
parser.add_argument('--norm', type=str, default='inf')
parser.add_argument('--eps', type=float, default=0.02)

args = parser.parse_args()

# # Load the chestx dataset

(x_train, y_train), (x_test, y_test), (mean_l2_train,
                                       mean_inf_train) = load_data(args.datapath, args.trainfile, args.testfile)

# # Create the model

sess, graph = create_model(args.weightspath, args.metaname, args.ckptname)

# # # Create the ART classifier

input_tensor = graph.get_tensor_by_name("input_1:0")
logit_tensor = graph.get_tensor_by_name("dense_3/MatMul:0")
output_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")
label_tensor = graph.get_tensor_by_name("dense_3_target:0")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logit_tensor, labels=label_tensor))

classifier = TFClassifier(input_ph=input_tensor, output=output_tensor,
                          labels_ph=label_tensor, loss=loss, sess=sess)

# # Generate adversarial examples

if args.norm == '2':
    norm = 2
    eps = mean_l2_train * args.eps
elif args.norm == 'inf':
    norm = np.inf
    eps = mean_inf_train * args.eps

noise_rand = random_sphere(nb_points=1,
                           nb_dims=(224 * 224 * 1),
                           radius=eps,
                           norm=norm)

noise_rand = noise_rand.reshape(224, 224, 1).astype('float32')
noise_rand = np.concatenate((noise_rand,) * 3, axis=-1)
noise_rand = noise_rand.astype(np.float32)
np.save('output/random_uap', noise_rand)

# # Evaluate the ART classifier on adversarial examples

acc_train, preds_train = get_preds(classifier, x_train, y_train)
acc_test, preds_test = get_preds(classifier, x_test, y_test)

x_train_adv_rand = x_train + noise_rand
x_test_adv_rand = x_test + noise_rand

acc_train_adv_rand, preds_train_adv_rand = get_preds(
    classifier, x_train_adv_rand, y_train)
acc_test_adv_rand, preds_test_adv_rand = get_preds(
    classifier, x_test_adv_rand, y_test)
fr_train_adv_rand = get_foolingrate_rate(preds_train_adv_rand, y_train)
fr_test_adv_rand = get_foolingrate_rate(preds_test_adv_rand, y_test)

rts_train_adv_rand_normal = get_target_success_rate(preds_train_adv_rand, 0)
rts_test_adv_rand_normal = get_target_success_rate(preds_test_adv_rand, 0)

rts_train_adv_rand_pneumonia = get_target_success_rate(preds_train_adv_rand, 1)
rts_test_adv_rand_pneumonia = get_target_success_rate(preds_test_adv_rand, 1)

rts_train_adv_rand_covid = get_target_success_rate(preds_train_adv_rand, 2)
rts_test_adv_rand_covid = get_target_success_rate(preds_test_adv_rand, 2)

print('=== train test ===')
print('acc {:.3f} {:.3f}'.format(acc_train, acc_test))
print('acc_adv_rand {:.3f} {:.3f}'.format(
    acc_train_adv_rand, acc_test_adv_rand))
print('fooling_rate_rand {:.3f} {:.3f}'.format(
    fr_train_adv_rand, fr_test_adv_rand))
print('target_success_rate_normal {:.3f} {:.3f}'.format(
    rts_train_adv_rand_normal, rts_test_adv_rand_normal))
print('target_success_rate_pneumonia {:.3f} {:.3f}'.format(
    rts_train_adv_rand_pneumonia, rts_test_adv_rand_pneumonia))
print('target_success_rate_covid {:.3f} {:.3f}'.format(
    rts_train_adv_rand_covid, rts_test_adv_rand_covid))

print('=== train ===')
show_confusion_matrix(np.argmax(y_train, axis=1), preds_train_adv_rand)
print('=== test ===')
show_confusion_matrix(np.argmax(y_test, axis=1), preds_test_adv_rand)

# # imshow
make_adv_img(x_test[0], noise_rand, x_test_adv_rand[0],
             'output/random_uap.png')