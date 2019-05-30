# http://www.xiaoliangbai.com/2016/09/09/more-on-image-noise-generation
# Source of the code is based on an excellent piece code from stackoverflow
# http://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

import cv2
import numpy as np
from scipy.ndimage import zoom
import os

angle1 = 30
angle2 = 60
angle3 = 90

gamma1 = 0.75  # 3/4
gamma2 = 0.65  # 13/20
gamma3 = 0.45  # 9/20
gamma4 = 0.25  # 1/4

kernel = (5, 5)

amount1 = 0.004
amount2 = 0.008
amount3 = 0.012

z1 = 1.1
z2 = 1.3
z3 = 1.5
z4 = 1.71
z5 = 1.9
z6 = 2.1


def add_zoom():
    global zoom1, zoom2, zoom3, zoom4, zoom5, zoom6
    zoom1 = clipped_zoom(img, z1)
    zoom2 = clipped_zoom(img, z2)
    zoom3 = clipped_zoom(img, z3)
    zoom4 = clipped_zoom(img, z4)
    zoom5 = clipped_zoom(img, z5)
    zoom6 = clipped_zoom(img, z6)


def add_perspective():
    global perspective1, perspective2, perspective3, perspective4, perspective5
    pts1 = np.float32([[cols, rows], [cols, 0], [0, rows], [0, 0]])
    pts2 = np.float32([[cols, 11 * rows / 12], [cols, rows / 12], [0, rows], [0, 0]])
    m1 = cv2.getPerspectiveTransform(pts1, pts2)
    perspective1 = cv2.warpPerspective(img, m1, (cols, rows))

    pts1 = np.float32([[cols, rows], [cols, 0], [0, rows], [0, 0]])
    pts2 = np.float32([[cols, 9 * rows / 10], [cols, rows / 10], [0, rows], [0, 0]])
    m2 = cv2.getPerspectiveTransform(pts1, pts2)
    perspective2 = cv2.warpPerspective(img, m2, (cols, rows))

    pts1 = np.float32([[cols, rows], [cols, 0], [0, rows], [0, 0]])
    pts2 = np.float32([[cols, 7 * rows / 8], [cols, rows / 8], [0, rows], [0, 0]])
    m3 = cv2.getPerspectiveTransform(pts1, pts2)
    perspective3 = cv2.warpPerspective(img, m3, (cols, rows))

    pts1 = np.float32([[cols, rows], [cols, 0], [0, rows], [0, 0]])
    pts2 = np.float32([[cols, 5 * rows / 6], [cols, rows / 6], [0, rows], [0, 0]])
    m4 = cv2.getPerspectiveTransform(pts1, pts2)
    perspective4 = cv2.warpPerspective(img, m4, (cols, rows))

    pts1 = np.float32([[cols, rows], [cols, 0], [0, rows], [0, 0]])
    pts2 = np.float32([[cols, 3 * rows / 4], [cols, rows / 4], [0, rows], [0, 0]])
    m5 = cv2.getPerspectiveTransform(pts1, pts2)
    perspective5 = cv2.warpPerspective(img, m5, (cols, rows))


def add_noise():
    global noise1, noise2, noise3
    noise1 = noise_generator('s&p', img1, amount1)
    noise2 = noise_generator('s&p', img2, amount2)
    noise3 = noise_generator('s&p', img3, amount3)


def add_rotation():
    global rotation1, rotation2, rotation3
    m1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle1, 1)
    rotation1 = cv2.warpAffine(img, m1, (cols, rows))

    m2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle2, 1)
    rotation2 = cv2.warpAffine(img, m2, (cols, rows))

    m3 = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle3, 1)
    rotation3 = cv2.warpAffine(img, m3, (cols, rows))


def add_gamma():
    global lighting1, lighting2, lighting3, lighting4
    lighting1 = adjust_gamma(img, gamma=gamma1)
    lighting2 = adjust_gamma(img, gamma=gamma2)
    lighting3 = adjust_gamma(img, gamma=gamma3)
    lighting4 = adjust_gamma(img, gamma=gamma4)


def add_gaussian_blur():
    global blur1, blur2, blur3, blur4
    blur1 = cv2.GaussianBlur(img,   kernel, 0)
    blur2 = cv2.GaussianBlur(blur1, kernel, 0)
    blur3 = cv2.GaussianBlur(blur2, kernel, 0)
    blur4 = cv2.GaussianBlur(blur3, kernel, 0)


def noise_generator(noise_type, image, amount):
    """
    Generate noise to a given Image based on required noise type

    Input parameters:
        image: ndarray (input image data. It will be converted to float)

        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row, col, ch = image.shape
    if noise_type == "gauss":
        mean = 0.0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = amount
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    else:
        return image


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


name_original = 'pferd_rechts0.JPG'
out_directory = 'output'

print(name_original)
print(out_directory)

img  = cv2.imread(name_original)
img1 = cv2.imread(name_original)
img2 = cv2.imread(name_original)
img3 = cv2.imread(name_original)
rows, cols, ch = img.shape

cv2.imwrite('original.jpg', img)

add_gaussian_blur()

cv2.imwrite(os.path.join(out_directory, 'blur1.jpg'), blur1)
cv2.imwrite(os.path.join(out_directory, 'blur2.jpg'), blur2)
cv2.imwrite(os.path.join(out_directory, 'blur3.jpg'), blur3)
cv2.imwrite(os.path.join(out_directory, 'blur4.jpg'), blur4)

add_gamma()

cv2.imwrite(os.path.join(out_directory, 'lighting1.jpg'), lighting1)
cv2.imwrite(os.path.join(out_directory, 'lighting2.jpg'), lighting2)
cv2.imwrite(os.path.join(out_directory, 'lighting3.jpg'), lighting3)
cv2.imwrite(os.path.join(out_directory, 'lighting4.jpg'), lighting4)

add_perspective()

cv2.imwrite(os.path.join(out_directory, 'perspective1.jpg'), perspective1)
cv2.imwrite(os.path.join(out_directory, 'perspective2.jpg'), perspective2)
cv2.imwrite(os.path.join(out_directory, 'perspective3.jpg'), perspective3)
cv2.imwrite(os.path.join(out_directory, 'perspective4.jpg'), perspective4)
cv2.imwrite(os.path.join(out_directory, 'perspective5.jpg'), perspective5)

add_rotation()

cv2.imwrite(os.path.join(out_directory, 'rotation1.jpg'), rotation1)
cv2.imwrite(os.path.join(out_directory, 'rotation2.jpg'), rotation2)
cv2.imwrite(os.path.join(out_directory, 'rotation3.jpg'), rotation3)

add_zoom()

cv2.imwrite(os.path.join(out_directory, 'zoom1.jpg'), zoom1)
cv2.imwrite(os.path.join(out_directory, 'zoom2.jpg'), zoom2)
cv2.imwrite(os.path.join(out_directory, 'zoom2.jpg'), zoom3)
cv2.imwrite(os.path.join(out_directory, 'zoom4.jpg'), zoom4)
cv2.imwrite(os.path.join(out_directory, 'zoom5.jpg'), zoom5)
cv2.imwrite(os.path.join(out_directory, 'zoom6.jpg'), zoom6)

add_noise()

cv2.imwrite(os.path.join(out_directory, 'noise1.jpg'), noise1)
cv2.imwrite(os.path.join(out_directory, 'noise2.jpg'), noise2)
cv2.imwrite(os.path.join(out_directory, 'noise3.jpg'), noise3)
