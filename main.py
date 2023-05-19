import glob
import numpy as np
import skimage
import torch
import random

if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('Using CPU')
    device = 'cpu'

rescale_factor = 0.25
inv_factor = int(1 / rescale_factor)
fragment_size = (256, 256)
limit = 0.5 * rescale_factor
pass1 = torch.abs(torch.fft.rfftfreq(fragment_size[-1])) < limit
pass2 = torch.abs(torch.fft.fftfreq(fragment_size[-2])) < limit
kernel = torch.outer(pass2, pass1).to(device=device)


def sk_resample(img):
    rescaled_img = skimage.transform.rescale(img, rescale_factor, anti_aliasing=True, channel_axis=0)
    return rescaled_img


def fft_resample(img):
    image = torch.from_numpy(img).to(device=device)
    transform = torch.fft.rfft2(image)
    transform[0] *= kernel
    transform[1] *= kernel
    transform[2] *= kernel
    rescaled_img = torch.fft.irfft2(transform)
    return rescaled_img[:, ::inv_factor, ::inv_factor]


def main(file_list, samples_per_file):
    images = []
    for path in file_list:
        # Reading all images, normalizing and converting to channel-first format
        images.append(np.moveaxis(np.float32(skimage.io.imread(path)) / 255.0, -1, 0))

    for image in images:
        for sample in range(samples_per_file):
            offset_x = random.randrange(0, image.shape[1] - 1 - fragment_size[0])
            offset_y = random.randrange(0, image.shape[2] - 1 - fragment_size[1])
            fragment = image[:, offset_x:offset_x + fragment_size[0], offset_y:offset_y + fragment_size[1]]

            sk_resample(fragment)
            fft_resample(fragment)


if __name__ == '__main__':
    # Searching all jpg files in test_files folder
    filelist = glob.glob('./test_files/*.jpg')
    main(filelist, samples_per_file=256)
