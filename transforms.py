import os

import numpy as np
import math
import cv2
import PIL
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps
import skimage
import subprocess
import tempfile

import torch
from torchvision.transforms.functional import gaussian_blur



BGR_FORMAT = 'BGR'
RGB_FORMAT = 'RGB'


def get_cv2_kernel_size(sigma, dtype):
    return round(sigma * (3 if dtype == np.uint8 else 4) * 2 + 1)
    

def flip_color_channels(image):
    return image[:, :, ::-1]


def make_grayscale(img, keep_channels=True, channel_axis=-1, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if keep_channels:
        img = skimage.color.gray2rgb(img, channel_axis=channel_axis)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)
    return img


def enhance(img, which, factor, format=BGR_FORMAT):
    assert which in ['color', 'contrast', 'brightness', 'sharpness']
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    if which == 'color':
        _Enhancer = ImageEnhance.Color
    elif which == 'contrast':
        _Enhancer = ImageEnhance.Contrast
    elif which == 'brightness':
        _Enhancer = ImageEnhance.Brightness
    elif which == 'sharpness':
        _Enhancer = ImageEnhance.Sharpness
    img = np.array(_Enhancer(PIL.Image.fromarray(img)).enhance(factor=factor))

    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return img


def unsharp_mask(img, radius=2.0, amount=1.5, threshold=3, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = np.array(PIL.Image.fromarray(img).filter(
        ImageFilter.UnsharpMask(radius=radius, percent=int(100 * amount), threshold=threshold)))

    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return img


def adjust_gamma(img, gamma=1, gain=1, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = skimage.exposure.adjust_gamma(image=img, gamma=gamma, gain=gain)

    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    return img


def adjust_log(img, gain=1, inv=False, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = skimage.exposure.adjust_log(image=img, gain=gain, inv=inv)

    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    return img


def adjust_sigmoid(img, cutoff=0.5, gain=10, inv=False, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = skimage.exposure.adjust_sigmoid(image=img, cutoff=cutoff, gain=gain, inv=inv)

    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    return img


def rescale_intensity(img, percentile=None, in_range='image', out_range='dtype', format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    assert percentile is None or in_range is None

    if percentile is not None:
        assert 0 < percentile < 50
        in_range = tuple(np.percentile(img, (percentile, 100 - percentile)))
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = skimage.exposure.rescale_intensity(image=img, in_range=in_range, out_range=out_range)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img_rescale = np.zeros_like(img)
        for i in [0, 1, 2]:
            img_rescale[:, :, i] = skimage.exposure.rescale_intensity(image=img[:, :, i], in_range=in_range, out_range=out_range)
        img = img_rescale
    else:
        raise ValueError

    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    return img


def adaptive_histogram_equalization(img, clip_limit=1, tile_grid_size=(8, 8), format=BGR_FORMAT):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    https://doi.org/10.3390/rs11111381
    """
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # img = skimage.exposure.equalize_adapthist(image=img, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
    # img = (img * 255).clip(0, 255).astype(np.uint8)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)
    return img


def histogram_equalization(img, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)
    return img


def autocontrast(img, cutoff=(0, 0), preserve_tone=False, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = np.array(ImageOps.autocontrast(PIL.Image.fromarray(img), cutoff=cutoff, preserve_tone=preserve_tone))

    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return img


def automatic_white_balance(img, method, ace_slope=10, ace_limit=1000, ace_samples=500, format=BGR_FORMAT):
    """

    :param img: image
    :param method: One of the following:
    GW - grey world (https://doi.org/10.1109/ISCE.2005.1502356)
    RX - retinex (https://doi.org/10.1109/ISCE.2005.1502356)
    GW_RX - quadratic combinic of grey world and retinex (https://doi.org/10.1109/ISCE.2005.1502356)
    ACE - automatic color equalization (https://doi.org/10.1016/S0167-8655(02)00323-9)
    :param ace_slope: only used method == "ACE"
    :param ace_limit: only used method == "ACE"
    :param ace_samples: only used method == "ACE"
    :param format: BGR or RGB format
    :return:
    """
    assert format in [BGR_FORMAT, RGB_FORMAT]
    assert method in ['GW', 'RX', 'GW_RX', 'ACE']

    if method in ['GW', 'RX']:
        # https://doi.org/10.1109/ISCE.2005.1502356
        # based on https://pypi.org/project/colorcorrect/
        # GREEN channel stays unchanged, hence both BGR and RGB will work seamlessly
        img = img.transpose(2, 0, 1).astype(np.uint32)
        agg_func = np.average if method == 'GW' else np.max
        coef_g = agg_func(img[1])
        for channel in [0, 2]:
            img[channel] = (img[channel] * (coef_g / agg_func(img[channel]))).clip(0, 255)
        img = img.transpose(1, 2, 0).astype(np.uint8)

    elif method == 'GW_RX':
        # https://doi.org/10.1109/ISCE.2005.1502356
        # based on https://pypi.org/project/colorcorrect/
        # GREEN channel stays unchanged, hence both BGR and RGB will work seamlessly
        img = img.transpose(2, 0, 1).astype(np.uint32)
        sum_g = np.sum(img[1])
        max_g = img[1].max()
        for channel in [0, 2]:
            _sum1 = np.sum(img[channel])
            _sum2 = np.sum(img[channel] ** 2)
            _max1 = img[channel].max()
            _max2 = _max1 ** 2
            coefficient = np.linalg.solve(np.array([[_sum2, _sum1], [_max2, _max1]]),
                                          np.array([sum_g, max_g]))
            img[channel] = ((img[channel] ** 2) * coefficient[0] + img[channel] * coefficient[1]).clip(0, 255)
        img = img.transpose(1, 2, 0).astype(np.uint8)

    elif method == 'ACE':
        # https://doi.org/10.1016/S0167-8655(02)00323-9
        # Must be RGB
        from colorcorrect.algorithm import automatic_color_equalization
        if format == BGR_FORMAT:
            img = flip_color_channels(img)
        img = automatic_color_equalization(img, slope=ace_slope, limit=ace_limit, samples=ace_samples)
        if format == BGR_FORMAT:
            img = flip_color_channels(img)

    return img


def dark_channel_prior_dehazing(img, dark_channels='BGR', percent=0.001, shift_blue_to_white=[False, False], format=BGR_FORMAT):
    """
    https://github.com/He-Zhang/image_dehaze
    https://www.sciencedirect.com/science/article/abs/pii/S1047320314001874
    shift_blue_to_white: https://arxiv.org/abs/1807.04169
    """

    def DarkChannel(im, sz, dark_channels='BGR', shift_blue_to_white=False):
        assert len(dark_channels) in [3, 2]

        if shift_blue_to_white:
            im = np.array(im)
            im[:, :, 1] = 1 - im[:, :, 1]
            im[:, :, 2] = 1 - im[:, :, 2]

        bgr = {l: c for l, c in zip(['B', 'G', 'R'], cv2.split(im))}
        dc = cv2.min(cv2.min(bgr['R'], bgr['G']), bgr['B']) if len(dark_channels) == 3 \
            else cv2.min(bgr[dark_channels[0]], bgr[dark_channels[1]])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)
        return dark

    def AtmLight(im, dark, percent=0.001, shift_blue_to_white=False):
        if not shift_blue_to_white:
            [h, w] = im.shape[:2]
            imsz = h * w
            numpx = int(max(math.floor(imsz * percent), 1))
            darkvec = dark.reshape(imsz);
            imvec = im.reshape(imsz, 3);

            indices = darkvec.argsort();
            indices = indices[imsz - numpx::]

            atmsum = np.zeros([1, 3])
            for ind in range(1, numpx):
                atmsum = atmsum + imvec[indices[ind]]

            A = atmsum / numpx;

        else:
            x = np.argmin(dark) // dark.shape[1]
            y = np.argmin(dark) % dark.shape[1]
            A = im[x, y, :].reshape((1, -1))

        return A

    def TransmissionEstimate(im, A, sz, dark_channels='BGR', shift_blue_to_white=False):
        omega = 0.95;
        im3 = np.empty(im.shape, im.dtype);

        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]

        transmission = 1 - omega * DarkChannel(im3, sz, dark_channels=dark_channels,
                                               shift_blue_to_white=shift_blue_to_white);
        return transmission

    def Guidedfilter(im, p, r, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
        mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
        cov_Ip = mean_Ip - mean_I * mean_p;

        mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
        var_I = mean_II - mean_I * mean_I;

        a = cov_Ip / (var_I + eps);
        b = mean_p - a * mean_I;

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

        q = mean_a * im + mean_b;
        return q;

    def TransmissionRefine(im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
        gray = np.float64(gray) / 255;
        r = 60;
        eps = 0.0001;
        t = Guidedfilter(gray, et, r, eps);

        return t;

    def Recover(im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype);
        t = cv2.max(t, tx);

        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

        return res

    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    I = img.astype('float64') / 255
    dark = DarkChannel(I, 15, dark_channels=dark_channels, shift_blue_to_white=shift_blue_to_white[0])
    A = AtmLight(I, dark, percent=percent, shift_blue_to_white=shift_blue_to_white[0])
    te = TransmissionEstimate(I, A, 15, dark_channels='BGR', shift_blue_to_white=shift_blue_to_white[1])
    t = TransmissionRefine(img, te)
    img = (Recover(I, t, A, 0.1) * 255).clip(0, 255).astype(np.uint8)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    return img


@DeprecationWarning
def unsupervised_colour_correction(img, format=BGR_FORMAT):
    def cal_equalisation(img, ratio):
        Array = img * ratio
        Array = np.clip(Array, 0, 255)
        return Array

    def RGB_equalisation(img):
        img = np.float32(img)
        avg_RGB = []
        for i in range(3):
            avg = np.mean(img[:, :, i])
            avg_RGB.append(avg)
        # print('avg_RGB',avg_RGB)
        a_r = avg_RGB[0] / avg_RGB[2]
        a_g = avg_RGB[0] / avg_RGB[1]
        ratio = [0, a_g, a_r]
        for i in range(1, 3):
            img[:, :, i] = cal_equalisation(img[:, :, i], ratio[i])
        return img

    def histogram_r(r_array, height, width):
        length = height * width
        R_rray = []
        for i in range(height):
            for j in range(width):
                R_rray.append(r_array[i][j])
        R_rray.sort()
        I_min = int(R_rray[int(length / 500)])
        I_max = int(R_rray[-int(length / 500)])
        array_Global_histogram_stretching = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                if r_array[i][j] < I_min:
                    # p_out = r_array[i][j]
                    array_Global_histogram_stretching[i][j] = I_min
                elif (r_array[i][j] > I_max):
                    p_out = r_array[i][j]
                    array_Global_histogram_stretching[i][j] = 255
                else:
                    p_out = int((r_array[i][j] - I_min) * ((255 - I_min) / (I_max - I_min))) + I_min
                    array_Global_histogram_stretching[i][j] = p_out
        return (array_Global_histogram_stretching)

    def histogram_g(r_array, height, width):
        length = height * width
        R_rray = []
        for i in range(height):
            for j in range(width):
                R_rray.append(r_array[i][j])
        R_rray.sort()
        I_min = int(R_rray[int(length / 500)])
        I_max = int(R_rray[-int(length / 500)])
        array_Global_histogram_stretching = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                if r_array[i][j] < I_min:
                    p_out = r_array[i][j]
                    array_Global_histogram_stretching[i][j] = 0
                elif (r_array[i][j] > I_max):
                    p_out = r_array[i][j]
                    array_Global_histogram_stretching[i][j] = 255
                else:
                    p_out = int((r_array[i][j] - I_min) * ((255) / (I_max - I_min)))
                    array_Global_histogram_stretching[i][j] = p_out
        return (array_Global_histogram_stretching)

    def histogram_b(r_array, height, width):
        length = height * width
        R_rray = []
        for i in range(height):
            for j in range(width):
                R_rray.append(r_array[i][j])
        R_rray.sort()
        I_min = int(R_rray[int(length / 500)])
        I_max = int(R_rray[-int(length / 500)])
        array_Global_histogram_stretching = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                if r_array[i][j] < I_min:
                    # p_out = r_array[i][j]
                    array_Global_histogram_stretching[i][j] = 0
                elif (r_array[i][j] > I_max):
                    # p_out = r_array[i][j]
                    array_Global_histogram_stretching[i][j] = I_max
                else:
                    p_out = int((r_array[i][j] - I_min) * ((I_max) / (I_max - I_min)))
                    array_Global_histogram_stretching[i][j] = p_out
        return (array_Global_histogram_stretching)

    def stretching(img):
        height = len(img)
        width = len(img[0])
        img[:, :, 2] = histogram_r(img[:, :, 2], height, width)
        img[:, :, 1] = histogram_g(img[:, :, 1], height, width)
        img[:, :, 0] = histogram_b(img[:, :, 0], height, width)
        return img

    def global_stretching(img_L, height, width):
        I_min = np.min(img_L)
        I_max = np.max(img_L)
        I_mean = np.mean(img_L)

        # print('I_min',I_min)
        # print('I_max',I_max)
        # print('I_max',I_mean)

        array_Global_histogram_stretching_L = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                p_out = (img_L[i][j] - I_min) * ((1) / (I_max - I_min))
                array_Global_histogram_stretching_L[i][j] = p_out

        return array_Global_histogram_stretching_L

    def HSVStretching(sceneRadiance):
        sceneRadiance = np.uint8(sceneRadiance)
        height = len(sceneRadiance)
        width = len(sceneRadiance[0])
        img_hsv = skimage.color.rgb2hsv(sceneRadiance)
        h, s, v = cv2.split(img_hsv)
        img_s_stretching = global_stretching(s, height, width)
        img_v_stretching = global_stretching(v, height, width)

        labArray = np.zeros((height, width, 3), 'float64')
        labArray[:, :, 0] = h
        labArray[:, :, 1] = img_s_stretching
        labArray[:, :, 2] = img_v_stretching
        img_rgb = skimage.color.hsv2rgb(labArray) * 255

        # img_rgb = np.clip(img_rgb, 0, 255)

        return img_rgb

    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    img = RGB_equalisation(img)
    img = stretching(img)
    img = HSVStretching(img)
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    return img


@DeprecationWarning
def multi_scale_retinex(img, method='MSRCR', sigma_list=[15, 80, 250], G=5.0, b=25.0, alpha=125.0, beta=46.0,
                        low_clip=0, high_clip=1, format=BGR_FORMAT):
    '''
    https://github.com/dongb5/Retinex
    https://doi.org/10.1109/83.597272
    '''

    def singleScaleRetinex(img, sigma):

        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

        return retinex

    def multiScaleRetinex(img, sigma_list):

        retinex = np.zeros_like(img)
        for sigma in sigma_list:
            retinex += singleScaleRetinex(img, sigma)

        retinex = retinex / len(sigma_list)

        return retinex

    def colorRestoration(img, alpha, beta):

        img_sum = np.sum(img, axis=2, keepdims=True)

        color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

        return color_restoration

    def simplestColorBalance(img, low_clip, high_clip):

        total = img.shape[0] * img.shape[1]
        for i in range(img.shape[2]):
            unique, counts = np.unique(img[:, :, i], return_counts=True)
            current = 0
            low_val = np.min(unique)
            high_val = np.max(unique)
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

        return img

    def _MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):

        img = np.float64(img) + 1.0

        img_retinex = multiScaleRetinex(img, sigma_list)
        img_color = colorRestoration(img, alpha, beta)
        img_msrcr = G * (img_retinex * img_color + b)

        for i in range(img_msrcr.shape[2]):
            img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                                 (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                                 255

        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

        return img_msrcr

    def automatedMSRCR(img, sigma_list):

        img = np.float64(img) + 1.0

        img_retinex = multiScaleRetinex(img, sigma_list)

        for i in range(img_retinex.shape[2]):
            unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
            for u, c in zip(unique, count):
                if u == 0:
                    zero_count = c
                    break

            low_val = unique[0] / 100.0
            high_val = unique[-1] / 100.0
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break

            img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

            img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                                   (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                                   * 255

        img_retinex = np.uint8(img_retinex)

        return img_retinex

    def MSRCP(img, sigma_list, low_clip, high_clip):

        img = np.float64(img) + 1.0

        intensity = np.sum(img, axis=2) / img.shape[2]

        retinex = multiScaleRetinex(intensity, sigma_list)

        intensity = np.expand_dims(intensity, 2)
        retinex = np.expand_dims(retinex, 2)

        intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

        intensity1 = (intensity1 - np.min(intensity1)) / \
                     (np.max(intensity1) - np.min(intensity1)) * \
                     255.0 + 1.0

        img_msrcp = np.zeros_like(img)

        for y in range(img_msrcp.shape[0]):
            for x in range(img_msrcp.shape[1]):
                B = np.max(img[y, x])
                A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
                img_msrcp[y, x, 0] = A * img[y, x, 0]
                img_msrcp[y, x, 1] = A * img[y, x, 1]
                img_msrcp[y, x, 2] = A * img[y, x, 2]

        img_msrcp = np.uint8(img_msrcp - 1.0)

        return img_msrcp

    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    if method == 'MSRCR':
        img = _MSRCR(
            img,
            sigma_list=sigma_list,
            G=G,
            b=b,
            alpha=alpha,
            beta=beta,
            low_clip=low_clip,
            high_clip=high_clip
        )
    elif method == 'auto_MSRCR':
        img = automatedMSRCR(
            img,
            sigma_list=sigma_list
        )
    elif method == 'MSRCP':
        img = MSRCP(
            img,
            sigma_list=sigma_list,
            low_clip=low_clip,
            high_clip=high_clip
        )
    elif method == 'MSR':
        img = np.float64(img) + 1.0
        img = multiScaleRetinex(
            img, sigma_list=sigma_list,
        )
        for i in range(img.shape[2]):
            img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / \
                           (np.max(img[:, :, i]) - np.min(img[:, :, i])) * \
                           255
        img = img.clip(0, 255).astype(np.uint8)
    elif method == 'SSR':
        img = np.float64(img) + 1.0
        img = singleScaleRetinex(
            img, sigma=sigma_list[1],
        )
        for i in range(img.shape[2]):
            img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / \
                           (np.max(img[:, :, i]) - np.min(img[:, :, i])) * \
                           255
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError

    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    return img


def variable_blur(im, sigma, ksize=None):
    """Blur an image with a variable Gaussian kernel.

    Parameters
    ----------
    im: numpy array, (h, w)

    sigma: numpy array, (h, w)

    ksize: int
        The box blur kernel size. Should be an odd number >= 3.

    Returns
    -------
    im_blurred: numpy array, (h, w)

    """

    if sigma == 15:
        blur_weight = 0.06147540983606558
        max_blurs = 3
        ksize = 121 if ksize is None else ksize
    elif sigma == 80:
        blur_weight = 0.06230529595015576
        max_blurs = 3
        ksize = 641 if ksize is None else ksize
    elif sigma == 250:
        blur_weight = 0.06243756243756243
        max_blurs = 3
        ksize = 2001 if ksize is None else ksize
    else:
        if ksize is None:
            ksize = round(sigma * (3 if im.dtype == np.uint8 else 4) * 2 + 1)
            ksize = max(ksize, 1)
            assert ksize > 0 and ksize % 2 == 1
        variance = box_blur_variance(ksize)
        # Number of times to blur per-pixel
        num_box_blurs = 2 * sigma ** 2 / variance
        # Number of rounds of blurring
        max_blurs = int(np.ceil(np.max(num_box_blurs))) * 3
        # Approximate blurring a variable number of times
        blur_weight = num_box_blurs / max_blurs

    current_im = im
    for i in range(max_blurs):
        next_im = cv2.blur(current_im, (ksize, ksize))
        current_im = next_im * blur_weight + current_im * (1 - blur_weight)
    return current_im


def box_blur_variance(ksize):
    x = np.arange(ksize) - ksize // 2
    x, y = np.meshgrid(x, x)
    return np.mean(x ** 2 + y ** 2)


def MSRCR(img, sigmas=[15, 80, 250], alpha=125.0, beta=46.0, G=5.0, b=25.0, flavour='MSRCR',
          approx=None, format=BGR_FORMAT, return_time=False):
    if return_time:
        import time

        def start_timing():
            return time.perf_counter()

        def elapsed_time(start):
            return time.perf_counter() - start

    assert approx in [None, 'faithful', 'fast', 'gpu']
    """
    '''
    https://github.com/adiMallya/retinex
    https://doi.org/10.1109/83.597272
    '''

    MSRCR (Multi-scale retinex with color restoration)

    Parameters :

    img : input image
    sigmas : list of all standard deviations in the X and Y directions, for Gaussian filter
    alpha : controls the strength of the nonlinearity
    beta : gain constant
    G : final gain
    b : offset
    """
    if approx == 'gpu':
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        from nvidia.dali.math import log10

        @pipeline_def()
        def msr_15_80_250_gpu_pipeline(image_dir):
            files, labels = fn.readers.file(file_root=image_dir)
            images = fn.decoders.image(files, device='mixed')
            images = images / 255 + 1
            msr = ((log10(images) - log10(fn.gaussian_blur(images, sigma=15, window_size=121))) + (
                        log10(images) - log10(fn.gaussian_blur(images, sigma=80, window_size=169))) + (
                               log10(images) - log10(fn.gaussian_blur(images, sigma=250, window_size=169)))) / 3
            return msr

        @pipeline_def()
        def msr_15_80_gpu_pipeline(image_dir):
            files, labels = fn.readers.file(file_root=image_dir)
            images = fn.decoders.image(files, device='mixed')
            images = images / 255 + 1
            msr = ((log10(images) - log10(fn.gaussian_blur(images, sigma=15, window_size=121))) + (
                        log10(images) - log10(fn.gaussian_blur(images, sigma=80, window_size=169)))) / 2
            return msr

    def singleScale(img, sigma, tmp_dir=None):
        """
        Single-scale Retinex

        Parameters :

        img : input image
        sigma : the standard deviation in the X and Y directions, for Gaussian filter
        """

        if approx == 'faithful':
            gb_img = cv2.imread(os.path.join(tmp_dir, f'sigma{sigma}.jpg'))
            if format == BGR_FORMAT:
                gb_img = flip_color_channels(gb_img)
            gb_img = skimage.img_as_float64(gb_img) + 1
        elif approx == 'fast':
            gb_img = np.zeros_like(img)
            for i in range(img.shape[2]):
                gb_img[:, :, i] = variable_blur(img[:, :, i], sigma)
        elif approx == 'gpu':
            raise NotImplementedError()
        else:
            # gb_img = cv2.GaussianBlur(img, (0, 0), sigma)
            assert sigma in [15, 80, 250]
            gb_img = cv2.GaussianBlur(img, (121, 121) if sigma == 15 else (169, 169) if sigma == 80 else (169, 169) if sigma == 250 else "ERROR", sigma)

        ssr = np.log10(img) - np.log10(gb_img)
        return ssr

    def multiScale(img, sigmas: list, tmp_dir=None):
        """
        Multi-scale Retinex

        Parameters :

        img : input image
        sigma : list of all standard deviations in the X and Y directions, for Gaussian filter
        """
        retinex = np.zeros_like(img)
        for s in sigmas:
            retinex += singleScale(img, s, tmp_dir=tmp_dir)
        msr = retinex / len(sigmas)
        return msr

    def crf(img, alpha, beta):
        """
        CRF (Color restoration function)

        Parameters :

        img : input image
        alpha : controls the strength of the nonlinearity
        beta : gain constant
        """
        img_sum = np.sum(img, axis=2, keepdims=True)

        color_rest = beta * (np.log10(alpha * img) - np.log10(img_sum))
        return color_rest

    print(flavour, sigmas)
    with tempfile.TemporaryDirectory() as tmp_dir:
        if approx == 'faithful':
            cv2.imwrite(os.path.join(tmp_dir, 'input.jpg'), img)
            for s in [sigmas[1]] if flavour == 'SSR' else sigmas:
                subprocess.call([
                    os.path.join(os.getenv("HOME"), 'work', 'tools', 'FastGaussianBlur', 'fastblur'),
                    os.path.join(tmp_dir, 'input.jpg'),
                    os.path.join(tmp_dir, f'sigma{s}.jpg'),
                    str(s)
                ])

        if return_time:
            s = start_timing()

        if flavour in ['MSRCR', 'MSR']:
            if approx == 'gpu':
                with open(os.path.join(tmp_dir, 'file_list.txt'), mode='w') as f:
                    print(os.path.join('img', 'input.jpg'), file=f)
                os.makedirs(os.path.join(tmp_dir, 'img'), exist_ok=True)
                cv2.imwrite(os.path.join(tmp_dir, 'img', 'input.jpg'), img)
                if list(sigmas) == [15, 80, 250]:
                    pipe_gpu = msr_15_80_250_gpu_pipeline(image_dir=tmp_dir, batch_size=1, num_threads=1, device_id=0)
                elif list(sigmas) == [15, 80]:
                    pipe_gpu = msr_15_80_gpu_pipeline(image_dir=tmp_dir, batch_size=1, num_threads=1, device_id=0)
                else:
                    raise NotImplementedError()
                pipe_gpu.build()
                img_msr = pipe_gpu.run()[0].as_cpu().at(0)
            else:
                if format == BGR_FORMAT:
                    img_msr = flip_color_channels(img)
                img_msr = skimage.img_as_float64(img_msr) + 1
                img_msr = multiScale(img_msr, sigmas=sigmas, tmp_dir=tmp_dir)

        if format == BGR_FORMAT:
            img = flip_color_channels(img)
        img = skimage.img_as_float64(img) + 1

        if flavour == 'MSRCR':
            img = G * (img_msr * crf(img, alpha=alpha, beta=beta) + b)
        elif flavour == 'MSR':
            img = img_msr
        elif flavour == 'SSR':
            img = singleScale(img, sigma=sigmas[1], tmp_dir=tmp_dir)
        else:
            raise ValueError

        for i in range(img.shape[2]):
            img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255
        if return_time:
            t = elapsed_time(s)

    img = np.uint8(np.minimum(np.maximum(img, 0), 255))

    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return (img, t) if return_time else img


@DeprecationWarning
def autobrightness(img, clip_hist_percent=1, format=BGR_FORMAT):
    """
    https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    """
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    return img


def grey_world_LAB(img, brightness_factor=1.0, format=BGR_FORMAT):
    """
    https://www.youtube.com/watch?v=Z0-iM37wseI
    https://github.com/bnsreenu/python_for_microscopists/tree/master/Tips_Tricks_45_white-balance_using_python
    https://github.com/bnsreenu/python_for_microscopists
    """
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for c in [1, 2]:
        avg = np.average(img[:, :, c])
        img[:, :, c] = img[:, :, c] - ((avg - 128) * (img[:, :, 0] / 255) * brightness_factor)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = img.clip(0, 255).astype(np.uint8)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)
    return img


IMG_TRANSFORMS = {
    'none': lambda x: x,
    'bw': make_grayscale,

    'adjust_gamma_down': lambda x: adjust_gamma(x, gamma=0.7),
    'adjust_gamma_up': lambda x: adjust_gamma(x, gamma=1.3),
    'adjust_log': adjust_log,
    'adjust_sigmoid': adjust_sigmoid,
    'rescale_intensity': rescale_intensity,
    'autocontrast': autocontrast,

    'eq_hist': histogram_equalization,
    'CLAHE': adaptive_histogram_equalization,

    'MSRCR': MSRCR,
    'MSR': lambda x: MSRCR(img=x, flavour='MSR'),
    'SSR': lambda x: MSRCR(img=x, flavour='SSR'),

    'SSR_50': lambda x: MSRCR(img=x, flavour='SSR', sigmas=[50, 50, 50]),
    'MSR_15_80': lambda x: MSRCR(img=x, flavour='MSR', sigmas=[15, 80]),

    'MSRCR_gpu': lambda x: MSRCR(img=x, flavour='MSRCR', approx='gpu'),
    'MSR_gpu': lambda x: MSRCR(img=x, flavour='MSR', approx='gpu'),
    'MSR_gpu_15_80': lambda x: MSRCR(img=x, flavour='MSR', approx='gpu', sigmas=[15, 80]),

    'MSRCR_fast': lambda x: MSRCR(img=x, flavour='MSRCR', approx='fast'),

    'MSRCR_approx': lambda x: MSRCR(img=x, flavour='MSRCR', approx='faithful'),
    'MSR_approx': lambda x: MSRCR(img=x, flavour='MSR', approx='faithful'),
    'SSR_approx': lambda x: MSRCR(img=x, flavour='SSR', approx='faithful'),

    'MSR_approx_15_80': lambda x: MSRCR(img=x, flavour='MSR', approx='faithful', sigmas=[15, 80]),
    'MSR_approx_80_250': lambda x: MSRCR(img=x, flavour='MSR', approx='faithful', sigmas=[80, 250]),
    'MSR_approx_50_150': lambda x: MSRCR(img=x, flavour='MSR', approx='faithful', sigmas=[50, 150]),
    'SSR_approx_15': lambda x: MSRCR(img=x, flavour='SSR', approx='faithful', sigmas=[15, 15, 15]),
    'SSR_approx_250': lambda x: MSRCR(img=x, flavour='SSR', approx='faithful', sigmas=[250, 250, 250]),
    'SSR_approx_150': lambda x: MSRCR(img=x, flavour='SSR', approx='faithful', sigmas=[150, 150, 150]),
    'SSR_approx_50': lambda x: MSRCR(img=x, flavour='SSR', approx='faithful', sigmas=[50, 50, 50]),
    'SSR_approx_100': lambda x: MSRCR(img=x, flavour='SSR', approx='faithful', sigmas=[100, 100, 100]),

    'SSR_gpu_50': lambda x: MSRCR(img=x, flavour='SSR', approx='gpu', sigmas=[50, 50, 50]),

    'MSRCR_with_gamma_up': lambda x: adjust_gamma(MSRCR(img=x), gamma=1.3),
    'DCP': dark_channel_prior_dehazing,
    'DCP_with_gamma_down': lambda x: adjust_gamma(dark_channel_prior_dehazing(img=x), gamma=0.7),

    'grey_world': grey_world_LAB,
    'retinex': lambda x: automatic_white_balance(img=x, method='RX'),
    'ace': lambda x: automatic_white_balance(img=x, method='ACE'),

    'sharpen': lambda x: enhance(img=x, which='sharpness', factor=2),
    'unsharp_mask': unsharp_mask,

    # 'more_color': lambda x: enhance(img=x, which='color', factor=1.3),
    # 'more_contrast': lambda x: enhance(img=x, which='contrast', factor=1.3),
    # 'more_brightness': lambda x: enhance(img=x, which='brightness', factor=1.3),

    # 'less_color': lambda x: enhance(img=x, which='color', factor=0.75),
    # 'less_contrast': lambda x: enhance(img=x, which='contrast', factor=0.75),
    # 'less_brightness': lambda x: enhance(img=x, which='brightness', factor=0.75),

    # offline: 'ARCR' 'CAP' 'MLLE' 'FunIE_GAN'
}