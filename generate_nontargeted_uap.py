import argparse

import numpy as np
import tensorflow as tf
from art.attacks import UniversalPerturbationRGB2Gray
from art.classifiers import TFClassifier

from uap_utils import (create_model, get_foolingrate_rate, get_preds,
                       load_data, make_adv_img, show_confusion_matrix)

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

adv_crafter = UniversalPerturbationRGB2Gray(
    classifier,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted': False, 'eps': 0.001},
    max_iter=15,
    eps=eps,
    norm=norm)

_ = adv_crafter.generate(x_train)
noise = adv_crafter.noise[0, :]
noise = noise.astype(np.float32)
np.save('output/nontargeted_uap', noise)

# # Evaluate the ART classifier on adversarial examples

acc_train, preds_train = get_preds(classifier, x_train, y_train)
acc_test, preds_test = get_preds(classifier, x_test, y_test)

x_train_adv = x_train + noise
x_test_adv = x_test + noise

acc_train_adv, preds_train_adv = get_preds(classifier, x_train_adv, y_train)
acc_test_adv, preds_test_adv = get_preds(classifier, x_test_adv, y_test)
fr_train_adv = get_foolingrate_rate(preds_train_adv, y_train)
fr_test_adv = get_foolingrate_rate(preds_test_adv, y_test)

print('=== train test ===')
print('acc {:.3f} {:.3f}'.format(acc_train, acc_test))
print('acc_adv {:.3f} {:.3f}'.format(acc_train_adv, acc_test_adv))
print('fooling_rate {:.3f} {:.3f}'.format(fr_train_adv, fr_test_adv))

print('=== train ===')
show_confusion_matrix(np.argmax(y_train, axis=1), preds_train_adv)
print('=== test ===')
show_confusion_matrix(np.argmax(y_test, axis=1), preds_test_adv)

# # imshow
make_adv_img(x_test[0], noise, x_test_adv[0], 'output/nontargeted_uap.png')
