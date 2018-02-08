"""
Extracted from https://github.com/yenchenlin/pix2pix-tensorflow
"""
import os
import scipy.misc
import PIL
import numpy as np

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def load_data(enhanced_image_path, original_image_path, flip=True, is_test=False):
    enhanced_image = imread(enhanced_image_path)
    original_image = imread(original_image_path)

    enhanced_image, original_image = pre_process_images(enhanced_image, original_image, flip=flip, is_test=is_test)

    enhanced_image = enhanced_image / 127.5 - 1.
    original_image = original_image / 127.5 - 1.

    image = np.concatenate((enhanced_image, original_image), axis=2)

    return image

def pre_process_images(enhanced_image, original_image, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        enhanced_image = scipy.misc.imresize(enhanced_image, [fine_size, fine_size])
        original_image = scipy.misc.imresize(original_image, [fine_size, fine_size])
    else:
        enhanced_image = scipy.misc.imresize(enhanced_image, [load_size, load_size])
        original_image = scipy.misc.imresize(original_image, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        enhanced_image = enhanced_image[h1:h1+fine_size, w1:w1+fine_size]
        original_image = original_image[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            enhanced_image = np.fliplr(enhanced_image)
            original_image = np.fliplr(original_image)

    return enhanced_image, original_image

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def inverse_transform(images):
    return (images + 1.) / 2.

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
