import os
import time
import torch
import torch.nn.functional as F
import models.models as m

from utils.utils import *
from omegaconf import OmegaConf
LAMBDA = 0.20  # source-separation coefficient

def flip_img(x):
    return torch.flip(x, dims=[3])

def l2n(x):
    return x / x.norm(dim=1, keepdim=True)


def load_surrogate_model():
    """ Load white-box and black-box models

    :return:
        face recognition and attribute recognition models
    """

    # Load pretrain white-box FR surrogate model
    fr_model = m.IR_152((112, 112))
    fr_model.load_state_dict(torch.load('./models/ir152.pth', map_location=device))
    fr_model.to(device)
    fr_model.eval()

    # Load pretrain white-box AR surrogate model
    ar_model = m.IR_152_attr_all()
    ar_model.load_state_dict(torch.load('./models/ir152_ar.pth', map_location=device))
    ar_model.to(device)
    ar_model.eval()

    return fr_model, ar_model

'''
    Obtain intermediate features by hooker
'''
layer_name = "ir_152.body.49"
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

gouts = []
def backward_hook(module, gin, gout):
    gouts.append(gout[0].data)
    return gin

def infer_fr_model(attack_img, victim_img, source_img, fr_model):
    def fused(img):
        feat = l2n(fr_model(img))
        feat_flip = l2n(fr_model(flip_img(img)))
        return l2n(feat + feat_flip)

    attack_img_feat = fused(attack_img)
    victim_img_feat = fused(victim_img).detach_()
    source_img_feat = fused(source_img).detach_()
    return attack_img_feat, victim_img_feat, source_img_feat

def infer_ar_model(attack_img, victim_img, source_img, ar_model):
    def fused(img):
        ar_model(img)
        f1 = activation[layer_name].clone()
        f1 = torch.flatten(f1).expand(1, f1.numel())
        ar_model(flip_img(img))
        f2 = activation[layer_name].clone()
        f2 = torch.flatten(f2).expand(1, f2.numel())
        return l2n(l2n(f1) + l2n(f2))

    attack_img_mid_feat = fused(attack_img)
    victim_img_mid_feat = fused(victim_img).detach_()
    source_img_mid_feat = fused(source_img).detach_()
    return attack_img_mid_feat, victim_img_mid_feat, source_img_mid_feat


def sibling_attack_facesm(attack_img, victim_img, fr_model, ar_model, config):
    """ Perform Sibling-Attack

    Args:
        :param attack_img:
            attacker face image
        :param victim_img:
            victim face image
        :param fr_model:
            face recognition model
        :param ar_model:
            attribute recognition model
        :param config:
            attacking configurations

    :return:
        adversarial face image
    """
    epochs = config.attack['outer_loops']
    alpha = config.attack['alpha']
    eps = config.attack['eps']
    INNER_MAX_EPOCH = config.attack['inner_loops']
    magic = config.attack['gamma']

    for layer in list(ar_model.named_modules()):
        if layer[0] == layer_name:
            fw_hook = layer[1].register_forward_hook(get_activation(layer_name))
            bw_hook = layer[1].register_backward_hook(backward_hook)

    ori_attack_img = attack_img.clone()
    for i in range(1, epochs+1):
        pre = time.time()
        if i % 2 == 0:
            INNER_LR = 1.0 / 255.0 * magic
            attack_img_tmp = attack_img.clone()
            attack_img_tmp_list = []

            for j in range(INNER_MAX_EPOCH):
                attack_img_tmp.requires_grad = True
                attack_img_feat, victim_img_feat, source_img_feat = infer_fr_model(
                    attack_img_tmp, victim_img, ori_attack_img, fr_model)  # CHANGED: +source_img arg, +3rd return
                fr_adv_loss = (1 - cos_simi(attack_img_feat, victim_img_feat)) \
                              + LAMBDA * cos_simi(attack_img_feat, source_img_feat)  # CHANGED: objective
                fr_model.zero_grad()
                fr_adv_loss.backward()
                sign_grad = attack_img_tmp.grad.sign()

                # core of PGD algorithm
                adv_img = attack_img_tmp - 1.0 * INNER_LR * sign_grad
                eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
                adv_img = torch.clamp(ori_attack_img + eta, min=-1, max=1).detach_()
                attack_img_tmp_list.append(adv_img)
                attack_img_tmp = adv_img.clone()

            while gouts:
                tensor = gouts.pop()
                tensor.detach_()

            AR_grad_temp_list = []
            for attack_img_tmp in attack_img_tmp_list:
                attack_img_tmp.requires_grad = True
                attack_img_mid_feat, victim_img_mid_feat, source_img_mid_feat = infer_ar_model(
                    attack_img_tmp, victim_img, ori_attack_img, ar_model)  # CHANGED: +source_img arg, +3rd return
                ar_adv_loss = (1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)) \
                              + LAMBDA * cos_simi(attack_img_mid_feat, source_img_mid_feat)  # CHANGED: objective
                ar_model.zero_grad()

                ar_adv_loss.backward()
                grad = attack_img_tmp.grad
                AR_grad_temp_list.append(grad.clone())
                attack_img_tmp.detach_()

            aggr_grad_pic = torch.zeros_like(attack_img)
            for AR_grad_temp in AR_grad_temp_list:
                aggr_grad_pic += AR_grad_temp

            # use aggregrated gradients
            attack_img = attack_img_tmp_list[-1].clone()
            attack_img.requires_grad = True
            # FIXED — backward flows through attack_img
            attack_img_mid_feat, victim_img_mid_feat, source_img_mid_feat = infer_ar_model(
                attack_img, victim_img, ori_attack_img, ar_model)
            ar_adv_loss = (1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)) \
                          + LAMBDA * cos_simi(attack_img_mid_feat, source_img_mid_feat)  # CHANGED: objective

            ar_model.zero_grad()
            ar_adv_loss.backward()

            w = 0.0001
            sign_grad = (attack_img.grad + w * aggr_grad_pic).sign()

            # core of PGD algorithm
            # use 1-step FR adv example as mid
            adv_img = attack_img - magic * alpha * sign_grad
            eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
            attack_img = torch.clamp(ori_attack_img + eta, min=-1, max=1).detach_()

            if VERBOSE:
                print("[Epoch-%d](FR-branch) AR loss: %f, time cost: %fs" % (i, ar_adv_loss.item(), time.time() - pre))
        else:
            INNER_LR = 1.0 / 255.0 * (1 - magic)
            attack_img_tmp = attack_img.clone()
            attack_img_tmp_list = []

            for j in range(INNER_MAX_EPOCH):
                attack_img_tmp.requires_grad = True
                attack_img_mid_feat, victim_img_mid_feat, source_img_mid_feat = infer_ar_model(
                    attack_img_tmp, victim_img, ori_attack_img, ar_model)  # CHANGED: +source_img arg, +3rd return
                ar_adv_loss = (1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)) \
                              + LAMBDA * cos_simi(attack_img_mid_feat, source_img_mid_feat)  # CHANGED: objective
                ar_model.zero_grad()

                ar_adv_loss.backward()
                sign_grad = attack_img_tmp.grad.sign()

                # core of PGD algorithm
                adv_img = attack_img_tmp - 1.0 * INNER_LR * sign_grad
                eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
                adv_img = torch.clamp(ori_attack_img + eta, min=-1, max=1).detach_()
                attack_img_tmp_list.append(adv_img)
                attack_img_tmp = adv_img.clone()

            while gouts:
                tensor = gouts.pop()
                tensor.detach_()

            FR_grad_temp_list = []
            for attack_img_tmp in attack_img_tmp_list:
                attack_img_tmp.requires_grad = True
                attack_img_feat, victim_img_feat, source_img_feat = infer_fr_model(
                    attack_img_tmp, victim_img, ori_attack_img, fr_model)  # CHANGED: +source_img arg, +3rd return
                fr_adv_loss = (1 - cos_simi(attack_img_feat, victim_img_feat)) \
                    + LAMBDA * cos_simi(attack_img_feat, source_img_feat)  # CHANGED: objective
                fr_model.zero_grad()
                fr_adv_loss.backward()

                grad = attack_img_tmp.grad
                FR_grad_temp_list.append(grad.clone())
                attack_img_tmp.detach_()

            aggr_grad_pic = torch.zeros_like(attack_img)
            for FR_grad_temp in FR_grad_temp_list:
                aggr_grad_pic += FR_grad_temp

            # use aggregrated gradients
            attack_img = attack_img_tmp_list[-1].clone()
            attack_img.requires_grad = True
            # FIXED
            attack_img_feat, victim_img_feat, source_img_feat = infer_fr_model(
                attack_img, victim_img, ori_attack_img, fr_model)
            fr_adv_loss = (1 - cos_simi(attack_img_feat, victim_img_feat)) \
                          + LAMBDA * cos_simi(attack_img_feat, source_img_feat)  # CHANGED: objective

            fr_model.zero_grad()
            fr_adv_loss.backward()

            w = 0.0001
            sign_grad = (attack_img.grad + w * aggr_grad_pic).sign()

            # core of PGD algorithm
            # use 1-step FR adv example as mid
            adv_img = attack_img - (1 - magic) * alpha * sign_grad
            eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
            attack_img = torch.clamp(ori_attack_img + eta, min=-1, max=1).detach_()

            if VERBOSE:
                print("[Epoch-%d](AR-branch) FR loss: %f, time cost: %fs" % (i, fr_adv_loss.item(), time.time() - pre))

    return attack_img

VERBOSE = True

def run_and_log(csv_path, img_dir, attack_fn, fr_model, ar_model, config, out_col, save_dir):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if out_col not in df.columns:
        df[out_col] = ''
    os.makedirs(save_dir, exist_ok=True)

    for idx, row in df.iterrows():
        existing = row.get(out_col, '')
        if pd.notna(existing) and str(existing).strip() and os.path.exists(str(existing)):
            print("skip existing:", idx, existing)
            continue

        src_path = os.path.join(img_dir, row['img1'])
        tgt_path = os.path.join(img_dir, row['img2'])
        print(idx, src_path, "========", tgt_path)

        attack_img = load_img(src_path, config).to(device)
        victim_img = load_img(tgt_path, config).to(device)
        adv_attack_img = attack_fn(attack_img, victim_img, fr_model, ar_model, config)

        save_path = os.path.join(save_dir,
            os.path.basename(tgt_path).split('.')[0] + '+' +
            os.path.basename(src_path).split('.')[0] + '.png')
        save_adv_img(adv_attack_img.cpu(), save_path, config)

        df.at[idx, out_col] = save_path
        df.to_csv(csv_path, index=False)
        print("saved:", save_path)

if __name__ == '__main__':
    config = OmegaConf.load('./configs/config.yaml')
    gpu = config.attack['gpu']
    dataset_name = config.dataset['dataset_name']
    device = torch.device('cuda:' + str(gpu))

    CSV_PATH = './datasets/pairs.csv'
    IMG_DIR = '/content/face_module/dataset_extractedfaces/celeba_pairs'

    fr_model, ar_model = load_surrogate_model()
    run_and_log(CSV_PATH, IMG_DIR, sibling_attack_facesm, fr_model, ar_model, config,
                out_col='sibling_attack_facesm',
                save_dir='./results_adv_images_facesm/')
