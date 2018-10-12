import os
import pdb
import glob
import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator


def adjustData(img, label, data, cnt, val='F'):
    ''' Adjust images and labels using data flag for network inputs

    Args:
        img   (np.array):   Augmented images
        label (np.array):   Augmented labels
        data  (str):        Data Flag
        cnt   (int):        Data Identification
        val   (str):        Data Folder Flag
    '''
    # To Save Augmented iamges and labels
    #for i in range(img.shape[0]):
    #    cv2.imwrite('./augment/'+val+'_'+ str(cnt+i)+'_img.jpg',img[i,:,:])
    #    cv2.imwrite('./augment/'+val+'_'+ str(cnt+i)+'_label.jpg',label[i,:,:])

    img = img / 255

    if data == 'complete':
        label[label < 25] = 0
        label[label >= 25] = 1
        label = np.concatenate(((-label)+1, label),axis=-1)

    elif data == 'core': #need to change
        l1 = (label>25).astype(np.uint8)
        l2 = (label<75).astype(np.uint8)
        label1 = np.logical_and(l1,l2).astype(np.uint8)
        l1 = (label>125).astype(np.uint8)
        l2 = (label<175).astype(np.uint8)
        label2 = np.logical_and(l1, l2).astype(np.uint8)
        label = np.logical_or(label1, label2).astype(np.uint8)
        label = np.concatenate(((-label)+1, label),axis=-1)

    elif data == 'enhancing':
        l1 = (label>125).astype(np.uint8)
        l2 = (label<175).astype(np.uint8)
        label = np.logical_and(l1, l2).astype(np.uint8)
        label = np.concatenate(((-label)+1, label),axis=-1)

    else:
        raise ValueError('data_flag ERROR!')

    return img, label



def dataset(args, mode='train',
            image_color_mode = "grayscale", label_color_mode = "grayscale",
            image_save_prefix  = "image", label_save_prefix  = "label",
            save_to_dir = None, target_size = (240,240), seed = 1):

    ''' Prepare dataset ( pre-processing + augmentation(optional) )

    Args:
        args (argparse):          Arguments parsered in command-lind
        mode (str):               Mode ('train', 'valid', 'test')
        image_color_mode (str):   Image color Mode Flag
        label_color_mode (str):   Label color Mode Flag
        image_save_prefix (str):  Prefix to use for filnames of saved images
        label_save_prefix (str):  Prefix to use for filename of saved labels
        save_to_dir (str):        Save directory
        target_size (tuple):      Target Size
        seed (int):               Seed value
    '''

    # Data Augmentation
    if mode == 'train':
        shuffle=True
        image_datagen = ImageDataGenerator(rotation_range=20,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.2,
                                           zoom_range=0.1)
                                           #brightness_range=[0.8,1.2])

        label_datagen = ImageDataGenerator(rotation_range=20,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.2,
                                           zoom_range=0.1)
    elif mode == 'test' or mode == 'valid':
        shuffle=False
        image_datagen = ImageDataGenerator()
        label_datagen = ImageDataGenerator()
    else:
        raise ValueError('dataset mode ERROR!')

    image_generator1 = image_datagen.flow_from_directory(
        args.image_root,
        classes = [args.image_folder1],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = args.batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        shuffle = shuffle,
        seed = seed)

    label_generator1 = label_datagen.flow_from_directory(
        args.label_root,
        classes = [args.label_folder1],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = args.batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = label_save_prefix,
        shuffle = shuffle,
        seed = seed)

    data_generator1 = zip(image_generator1, label_generator1)
    cnt = 0

    if mode == 'test' or mode == 'valid':
        for (img,label) in data_generator1:
            img,label = adjustData(img, label, args.data, cnt)
            cnt+=img.shape[0]
            yield (img,label)

    else:
        image_generator2 = image_datagen.flow_from_directory(
            args.image_root,
            classes = [args.image_folder2],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = args.batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = image_save_prefix,
            shuffle = shuffle,
            seed = seed+1)

        label_generator2 = label_datagen.flow_from_directory(
            args.label_root,
            classes = [args.label_folder2],
            class_mode = None,
            color_mode = label_color_mode,
            target_size = target_size,
            batch_size = args.batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = label_save_prefix,
            shuffle = shuffle,
            seed = seed+1)
        data_generator2 = zip(image_generator2, label_generator2)

        while(True):
            if np.random.randint(3) == 2:
                for (img,label) in data_generator1:
                    img,label = adjustData(img, label, args.data, cnt, 'F')
                    cnt+=img.shape[0]
                    yield (img,label)

            else:
                for (img,label) in data_generator2:
                    img,label = adjustData(img, label, args.data, cnt, 'S')
                    cnt+=img.shape[0]
                    yield (img,label)


def save_result(args, output):
    ''' Save Prediction results overlapping on original MRI images

    Args:
        args (argparse):    Arguments parsered in command-lind
        output (np.array):  Prediction Results by segementation model
    '''

    file_path = []
    file_path += glob.glob(os.path.join(args.image_root, args.image_folder1, '*.jpg'))
    file_path = sorted(file_path)

    output = np.argmax(output, axis=-1)*255
    kernel = np.ones((5,5),np.uint8)

    for i in range(output.shape[0]):
        save_path = os.path.join(args.output_root, str(i)+'.jpg')

        img = cv2.imread(file_path[i])
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
        cv2.imwrite(save_path, img)
