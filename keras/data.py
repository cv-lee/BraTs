import os
import pdb
import numpy as np
import glob
import cv2

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans


def adjustData(img, label, data_flag, cnt):
    img = img / 255
    if data_flag == 'complete':
        label[label < 25] = 0
        label[label >= 25] = 1
        #for i in range(img.shape[0]):
        #    cv2.imwrite(str(cnt+i)+'_img.jpg',img[i,:,:]*255)
        #    cv2.imwrite(str(cnt+i)+'_label.jpg',label[i,:,:]*255)
        label = np.concatenate(((-label)+1, label),axis=-1)
    elif data_flag == 'core': #need to change
        l1 = (label>25).astype(np.uint8)
        l2 = (label<75).astype(np.uint8)
        label1 = np.logical_and(l1,l2).astype(np.uint8)
        l1 = (label>125).astype(np.uint8)
        l2 = (label<175).astype(np.uint8)
        label2 = np.logical_and(l1, l2).astype(np.uint8)
        label = np.logical_or(label1, label2).astype(np.uint8)
        label = np.concatenate(((-label)+1, label),axis=-1)
    elif data_flag == 'enhancing': # need to change
        l1 = (label>125).astype(np.uint8)
        l2 = (label<175).astype(np.uint8)
        label = np.logical_and(l1, l2).astype(np.uint8)
        label = np.concatenate(((-label)+1, label),axis=-1)
    else:
        raise ValueError('data_flag ERROR!')
    return (img, label)


def trainGenerator(args,
                   image_color_mode = "grayscale", label_color_mode = "grayscale",
                   image_save_prefix  = "image", label_save_prefix  = "label",
                   data_flag = 'complete', save_to_dir = None,
                   target_size = (240,240), seed = 1):

    img_aug = dict(
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=1, #need to change
            height_shift_range=1, #need to change
            shear_range=0.1,
            zoom_range=0.1,
            brightness_range=[0.8,1.2]# elastic distortion add
            )
    label_aug = dict(
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=1, #need to change
            height_shift_range=1, #need to change
            shear_range=0.1,
            zoom_range=0.1,
            )

    image_datagen = ImageDataGenerator(img_aug)
    label_datagen = ImageDataGenerator(label_aug)

    image_generator = image_datagen.flow_from_directory(
        args.train_root,
        classes = [args.image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = args.batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    label_generator = label_datagen.flow_from_directory(
        args.train_root,
        classes = [args.label_folder],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = args.batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = label_save_prefix,
        seed = seed)

    train_generator= zip(image_generator, label_generator)

    cnt=0
    for (img,label) in train_generator:
        img,label = adjustData(img, label, data_flag, cnt)
        cnt+=img.shape[0]
        yield (img,label)


def testGenerator(args, image_color_mode = "grayscale", label_color_mode = "grayscale",
                  image_save_prefix  = "image", label_save_prefix  = "mask",
                  data_flag = 'complete', save_to_dir = None, target_size = (240,240),
                  seed=1):

    image_datagen = ImageDataGenerator()
    label_datagen = ImageDataGenerator()

    image_generator = image_datagen.flow_from_directory(
        args.test_img_root,
        classes = [args.image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = args.batch_size,
        shuffle=False,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    label_generator = label_datagen.flow_from_directory(
        args.test_label_root,
        classes = [args.label_folder],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = args.batch_size,
        shuffle=False,
        save_to_dir = save_to_dir,
        save_prefix  = label_save_prefix,
        seed = seed)
    test_generator = zip(image_generator, label_generator)
    cnt = 0
    for (img,label) in test_generator:
        img,label = adjustData(img, label, data_flag, cnt)
        cnt+= img.shape[0]
        yield (img,label)


def save_img(args, output):
    file_path = []
    file_path += glob.glob(os.path.join(args.test_img_root, args.image_folder, '*.jpg'))
    file_path = sorted(file_path)
    output = np.argmax(output, axis=-1)*255
    kernel = np.ones((5,5),np.uint8)
    for i in range(output.shape[0]):
        img = cv2.imread(file_path[i])

        output_path = os.path.join(args.output_root, str(i)+'.jpg')
        pred = cv2.morphologyEx(output[i].astype(np.uint8), cv2.MORPH_OPEN, kernel)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
        pred = np.expand_dims(pred, axis=2)
        zeros = np.zeros(pred.shape)
        pred = np.concatenate((zeros,zeros,pred), axis=2)
        img = img + pred
        if img.max() > 0:
            img = (img/img.max())*255
        else:
            img = (img/1)*255
        cv2.imwrite(output_path, img)
