import argparse

import numpy as np
from art.attacks import TargetedUniversalPerturbationRGB2Gray

from uap_utils import (get_preds, get_target_success_rate, make_adv_img,
                       set_up, show_confusion_matrix)

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
parser.add_argument('--target', type=str, default='COVID-19')

args = parser.parse_args()
(x_train, y_train), (x_test, y_test), (mean_l2_train,
                                       mean_inf_train), norm, eps, classifier = set_up(args)

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
target = mapping[args.target]

# # Generate adversarial examples

adv_crafter = TargetedUniversalPerturbationRGB2Gray(
    classifier,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted': True, 'eps': 0.001},
    max_iter=15,
    eps=eps,
    norm=norm)

y_train_adv_tar = np.zeros(y_train.shape)
y_train_adv_tar[:, target] = 1.0
_ = adv_crafter.generate(x_train, y=y_train_adv_tar)
noise = adv_crafter.noise[0, :]
noise = noise.astype(np.float32)
np.save('output/targeted_uap_{}'.format(args.target), noise)

# # Evaluate the ART classifier on adversarial examples

acc_train, preds_train = get_preds(classifier, x_train, y_train)
acc_test, preds_test = get_preds(classifier, x_test, y_test)

x_train_adv = x_train + noise
x_test_adv = x_test + noise

acc_train_adv, preds_train_adv = get_preds(classifier, x_train_adv, y_train)
acc_test_adv, preds_test_adv = get_preds(classifier, x_test_adv, y_test)
rts_train_adv = get_target_success_rate(preds_train_adv, target)
rts_test_adv = get_target_success_rate(preds_test_adv, target)

print('=== train test ===')
print('acc {:.3f} {:.3f}'.format(acc_train, acc_test))
print('acc_adv {:.3f} {:.3f}'.format(acc_train_adv, acc_test_adv))
print('target_success_rate {:.3f} {:.3f}'.format(rts_train_adv, rts_test_adv))

print('=== train ===')
show_confusion_matrix(np.argmax(y_train, axis=1), preds_train_adv)
print('=== test ===')
show_confusion_matrix(np.argmax(y_test, axis=1), preds_test_adv)

# # imshow

make_adv_img(x_test[0], noise, x_test_adv[0],
             'output/targeted_uap_{}.png'.format(args.target))
