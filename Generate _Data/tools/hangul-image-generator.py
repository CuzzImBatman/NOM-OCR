#!/usr/bin/env python

import argparse
import glob
import io
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import skimage.filters as filters
import skimage.util
from skimage.transform import resize as sk_resize
from skimage import color
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                 # '../labels/2350-common-hangul.txt')
                                  '../labels/alphabetTK.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../image-data')

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 15

# Width and height of the resulting image.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_RESIZE = 64

def _resize_(image, size=IMAGE_RESIZE):
    image_size = image.shape[0]
    if (image_size < size):
        return image

    scale = size / image.shape[0]  # working only with square images
    sigma = (1 - scale) / 2.0
    image = filters.gaussian(image, sigma=sigma, preserve_range=True)
    image = sk_resize(image, (size, size), preserve_range=True)
    return image


def _invert_make_sparse_(image):
    image = skimage.util.invert(image)  #
    threshold = filters.threshold_otsu(image)
    threshold_index = image > threshold
    image[threshold_index] = 255
    return image


def _threshold_(image):
    final_threshold_index = image > 10
    image[final_threshold_index] = 255
    image[~final_threshold_index] = 0
    image = image.astype(np.float32) / 255
    image = np.ceil(image)
    return image


def add_salt_pepper_noise(orig_image, threshold_prob):
    image = orig_image.copy()
    max = np.max(image)
    min = np.min(image)
    random_image = np.random.rand(*image.shape)
    #random_image= image
    salt_index = random_image >= 1 - threshold_prob
    pepper_index = random_image < threshold_prob
    image[salt_index] = max
    image[pepper_index] = min
    return image

def invert (image):
    mtx= np.asarray(image)
    mtx = cv2.bitwise_not(mtx)
    invert = Image.fromarray(mtx)
    return invert
    

def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    """
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'hangul-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))
    #fonts = glob.glob(os.path.join(fonts_dir, '*.otf'))

    labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
                         encoding='utf-8')
    total_count = 0
    prev_count = 0
    for character in labels:
        char_dir = os.path.join(output_dir,(character))
        if not os.path.exists(char_dir):
            os.makedirs(os.path.join(char_dir))
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))
            
        for font in fonts:
            total_count += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
            font = ImageFont.truetype(font, 48)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize( chr(int(character)), font=font)
            drawing.text(
                ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                chr(int(character)),
                fill=(255),
                font=font
            )
            file_string = 'hangul_{}.jpeg'.format(total_count)
            file_path = os.path.join(char_dir, file_string)
            #image = 
            invert_image = invert(image)
            invert_image.save(file_path, 'JPEG')
            mtx= np.asarray(image)
            #save= _invert_make_sparse_(mtx)
            #save = Image.fromarray(save)
            #file_string = 'hangul_{}_sparse.jpeg'.format(total_count)
            #file_path = os.path.join(char_dir, file_string)
            #save=save.rotate(np.random.random_sample()*(14)-7, fillcolor = 255)
            #save.save(file_path, 'JPEG')
            add= add_salt_pepper_noise(mtx,0.015)
            add= Image.fromarray(add)
            add=invert(add)
            file_string = 'hangul_{}_noise.jpeg'.format(total_count)
            file_path = os.path.join(char_dir, file_string)
            add=add.rotate(np.random.random_sample()*(12)-6, fillcolor = 255)
            add.save(file_path, 'JPEG')
            
            #labels_csv.write(u'{},{}\n'.format(file_path,str(int(character))))
            
            for i in range(DISTORTION_COUNT):
                total_count += 1
                file_string = 'hangul_{}.jpeg'.format(total_count)
                file_path = os.path.join(char_dir, file_string)
                arr = np.array(image)

                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                distorted_image = Image.fromarray(distorted_array)
                
                invert_image = invert(distorted_image)
                invert_image=invert_image.rotate(np.random.random_sample()*(12)-6, fillcolor = 255)
                invert_image.save(file_path, 'JPEG')
                #mtx= np.asarray(image)
                save= _invert_make_sparse_(distorted_array)
                save = Image.fromarray(save)
                file_string = 'hangul_{}_sparse.jpeg'.format(total_count)
                file_path = os.path.join(char_dir, file_string)
                save=save.rotate(np.random.random_sample()*(12)-6, fillcolor = 255)
                save.save(file_path, 'JPEG')
                add= add_salt_pepper_noise(distorted_array,0.015)
                add= Image.fromarray(add)
                add= invert(add)
                file_string = 'hangul_{}_noise.jpeg'.format(total_count)
                file_path = os.path.join(char_dir, file_string)
                add=add.rotate(np.random.random_sample()*(12)-6, fillcolor = 255)
                add.save(file_path, 'JPEG')
                
                #labels_csv.write(u'{},{}\n'.format(file_path, str(int(character))))

    print('Finished generating {} images.'.format(total_count))
    labels_csv.close()


def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.

    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)
