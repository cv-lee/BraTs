import pdb
import cv2
import os
import numpy as np
import nibabel as nib
import torch
import sys
import time
import logging
import logging.handlers
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral,\
         create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def dice_coef(preds, targets, backprop=True):
    smooth = 1.0
    class_num = 2
    if backprop:
        for i in range(class_num):
            pred = preds[:,i,:,:]
            target = targets[:,i,:,:]
            intersection = (pred * target).sum()
            loss_ = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            if i == 0:
                loss = loss_
            else:
                loss = loss + loss_
        loss = loss/class_num
        return loss
    else:
        # Need to generalize
        targets = np.array(targets.argmax(1))
        if len(preds.shape) > 3:
            preds = np.array(preds).argmax(1)
        for i in range(class_num):
            pred = (preds==i).astype(np.uint8)
            target= (targets==i).astype(np.uint8)
            intersection = (pred * target).sum()
            loss_ = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            if i == 0:
                loss = loss_
            else:
                loss = loss + loss_
        loss = loss/class_num
        return loss


def get_crf_img(inputs, outputs):
    for i in range(outputs.shape[0]):
        img = inputs[i]
        softmax_prob = outputs[i]
        unary = unary_from_softmax(softmax_prob)
        unary = np.ascontiguousarray(unary)
        d = dcrf.DenseCRF(img.shape[0] * img.shape[1], 2)
        d.setUnaryEnergy(unary)
        feats = create_pairwise_gaussian(sdims=(10,10), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(sdims=(50,50), schan=(20,20,20),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)
        res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
        if i == 0:
            crf = np.expand_dims(res,axis=0)
        else:
            res = np.expand_dims(res,axis=0)
            crf = np.concatenate((crf,res),axis=0)
    return crf


def erode_dilate(outputs, kernel_size=7):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    outputs = outputs.astype(np.uint8)
    for i in range(outputs.shape[0]):
        img = outputs[i]
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        outputs[i] = img
    return outputs


def post_process(args, inputs, outputs, input_path=None,
                 crf_flag=True, erode_dilate_flag=True,
                 save=True, overlap=True):
    inputs = (np.array(inputs.squeeze()).astype(np.float32)) * 255
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs,inputs,inputs), axis=3)
    outputs = np.array(outputs)

    # Conditional Random Field
    if crf_flag:
        outputs = get_crf_img(inputs, outputs)
    else:
        outputs = outputs.argmax(1)

    # Erosion and Dilation
    if erode_dilate_flag:
        outputs = erode_dilate(outputs, kernel_size=7)
    if save == False:
        return outputs

    outputs = outputs*255
    for i in range(outputs.shape[0]):
        path = input_path[i].split('/')
        output_folder = os.path.join(args.output_root, path[-2])
        try:
            os.mkdir(output_folder)
        except:
            pass
        output_path = os.path.join(output_folder, path[-1])
        if overlap:
            img = outputs[i]
            img = np.expand_dims(img, axis=2)
            zeros = np.zeros(img.shape)
            img = np.concatenate((zeros,zeros,img), axis=2)
            img = np.array(img).astype(np.float32)
            img = inputs[i] + img
            if img.max() > 0:
                img = (img/img.max())*255
            else:
                img = (img/1) * 255
            cv2.imwrite(output_path, img)
        else:
            img = outputs[i]
            cv2.imwrite(output_path, img)
    return None

'''
TODO: Need to fix
def save_img(args, inputs, outputs, input_paths, overlap=True):
    inputs = (np.array(inputs.squeeze()).astype(np.float32)) * 255
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs,inputs,inputs), axis=3)
    inputs = np.expand_dims(inputs, axis=3)
    outputs = np.array(outputs.max(1)[1])*255
    kernel = np.ones((5,5),np.uint8)

    for i, path in enumerate(input_paths):
        path = path.split('/')[-2]
        if i == 0:
            compare = path
        else:
            if compare != path:
                raise ValueError('Output Merge Fail')
            pass

    final_img = None
    output_path = os.path.join(args.output_root, path+'.nii.gz')
    for i in range(outputs.shape[0]):
        if overlap:
            img = cv2.morphologyEx(outputs[i].astype(np.uint8), cv2.MORPH_OPEN, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = np.expand_dims(img, axis=2)
            zeros = np.zeros(img.shape)
            img = np.concatenate((zeros,zeros,img), axis=2)
            img = np.expand_dims(img, axis=2)
            img = np.array(img).astype(np.float32)
            img = inputs[i] + img
            if img.max() > 0:
                img = (img/img.max())*255
            else:
                img = (img/1) * 255
            img = np.expand_dims(img, axis=3)
            if i == 0:
                final_img = img
            else:
                final_img = np.concatenate((final_img,img),axis=3)
        else:
            img = output[i]
    output_path = os.path.join(args.output_root, path)
    final_img = nib.Nifti1Pair(final_img, np.eye(4))
    nib.save(final_img, output_path)
    print(output_path)
'''


class Checkpoint:
    def __init__(self, model, optimizer=None, epoch=0, best_score=1):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.best_score = best_score

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.epoch = checkpoint["epoch"]
        self.best_score = checkpoint["best_score"]
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in self.optimizer.state.values():
                  for k, v in state.items():
                           if torch.is_tensor(v):
                                    state[k] = v.cuda()

    def save(self, path):
        state_dict = self.model.module.state_dict()
        torch.save({"model_state": state_dict,
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_score": self.best_score}, path)


def progress_bar(current, total, msg=None):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_logger(level="DEBUG", file_level="DEBUG"):
    logger = logging.getLogger(None)
    logger.setLevel(level)
    fomatter = logging.Formatter(
            '%(asctime)s  [%(levelname)s]  %(message)s  (%(filename)s:  %(lineno)s)')
    fileHandler = logging.handlers.TimedRotatingFileHandler(
            'result.log', when='d', encoding='utf-8')
    fileHandler.setLevel(file_level)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)
    return logger
