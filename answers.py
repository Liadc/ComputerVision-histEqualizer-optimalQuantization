from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    if representation != 1 and representation != 2:
        raise ValueError('only 1 or 2 are possible inputs for representation, '
                         'to output Grayscale or RGB pics respectively')
    filepath = './pics/' + filename
    print(filepath)
    if representation == 1:
        channel = 0
    elif representation == 2:
        channel = 1
    img = cv2.imread(filepath, channel)
    if img is None:
        raise ValueError("Could not find requested file inside 'pics' folder. ")
    if channel:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return img


def imDisplay(filename: str, representation: int):
    img = imReadAndConvert(filename, representation)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.ceil(img)
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Liad and Timor showing image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transformRGB2YIQ(imRGB: np.ndarray) -> np.ndarray:
    mat = np.array([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    for x in imRGB:
        for y in x:
            y[:] = mat.dot(y[:])
    return imRGB


# get normlized (0,1) image
def transformYIQ2RGB(imYIQ: np.ndarray) -> np.ndarray:
    # mat = np.array([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    mat = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]])
    # mat = np.linalg.inv(mat)
    print(mat)
    for x in imYIQ:
        for y in x:
            y[:] = mat.dot(y[:])
    return imYIQ


def histogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    global histOrig, histEq

    if len(imOrig.shape) == 2:
        greyscale = True
    elif len(imOrig.shape) == 3:
        greyscale = False
    else:
        raise ValueError('Unsupported array representation. only RGB or Greyscale images allowed')
    imEq = np.copy(imOrig)
    # if greyscale:
        # imEq = cv2.normalize(imEq, None, 0, 255, cv2.NORM_MINMAX)
        # imEq = np.ceil(imEq)  # floating point precision correction
        # imEq = imEq.astype('uint8')
    if not greyscale:
        imEqT = cv2.cvtColor(imEq, cv2.COLOR_BGR2RGB)
        imEqT = cv2.normalize(imEqT.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        imEqT = transformRGB2YIQ(imEqT)  # imeqT is now YIQ
        imEq = imEqT[:, :, 0]
        imEq = cv2.normalize(imEq, None, 0, 255, cv2.NORM_MINMAX)
        imEq = np.ceil(imEq)  # floating point precision correction
        imEq = imEq.astype('uint8')

        # imEqT = transformRGB2YIQ(imEqT)
        # imEqT = cv2.normalize(imEqT, None, 0, 255, cv2.NORM_MINMAX)
        # imEqT = np.ceil(imEqT)  # floating point precision correction
        # imEqT = imEqT.astype('uint8')
        # imEq = imEqT[:, :, 0]

    # Original Histogram:
    plt.subplot(2, 1, 1)
    histOrig, bins = np.histogram(imEq.flatten(), 256, [0, 255])
    cdf = histOrig.cumsum()  # cumulative
    cdf_normalized = cdf * histOrig.max() / cdf.max()
    plt.title('Original image histogram with CDF (Liad & Timor)')
    plt.plot(cdf_normalized, color='b')
    plt.hist(imEq.flatten(), 256, [0, 255], color='r')
    plt.xlim([0, 255])
    plt.legend(('cdf - ORIGINAL', 'histogram - ORIGINAL'), loc='upper left')
    # plt.show()

    # equalize operation on the copy of original image
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    imEq = cdf[imEq]

    # histogram for equalized image:
    histEq, bins = np.histogram(imEq.flatten(), 256, [0, 255])
    cdf = histEq.cumsum()  # cumulative
    cdf_normalized = cdf * histEq.max() / cdf.max()
    plt.subplot(2, 1, 2)
    plt.title('Equalized image histogram with CDF (Liad & Timor)')
    plt.plot(cdf_normalized, color='b')
    plt.hist(imEq.flatten(), 256, [0, 255], color='r')
    plt.xlim([0, 255])
    plt.legend(('cdf - EQUALIZED', 'histogram - EQUALIZED'), loc='upper right')
    plt.show()

    # display original image
    cv2.imshow('Liad Timor and Moshe showing the ORIGINAL image', imOrig)
    if not greyscale:
        imEq = cv2.normalize(imEq.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        imEqT[:, :, 0] = imEq
        imEq = transformYIQ2RGB(imEqT)
        imEq = cv2.normalize(imEq, None, 0, 255, cv2.NORM_MINMAX)
        imEq = np.ceil(imEq)
        imEq = imEq.astype('uint8')
        imEq = cv2.cvtColor(imEq, cv2.COLOR_RGB2BGR)
    # display equalized image
    cv2.imshow('Liad and Timor showing the EQUALIZED image', imEq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imEq, histOrig, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    if len(imOrig.shape) is 2:
        imQ = imOrig
        imQ = cv2.normalize(imQ, None, 0, 255, cv2.NORM_MINMAX)
        imQ = np.ceil(imQ)
    elif len(imOrig.shape) is 3:
        imT = transformRGB2YIQ(imOrig)
        imQ = imT[:,:,0]

    imQ = cv2.normalize(imQ, None, 0, 255, cv2.NORM_MINMAX)
    imQ = np.ceil(imQ)
    imQ = imQ.astype('uint8')

    pass


try:
    img = cv2.imread('./pics/bla2.jpg', 1)
    a, b, c = histogramEqualize(img)

    # test transform:
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # img = transformRGB2YIQ(img)
    # img = transformYIQ2RGB(img)
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # img = np.ceil(img)
    # img = img.astype('uint8')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Liad and Timor showing the EQUALIZED image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


except ValueError as err:
    print(err.args)

cv2.destroyAllWindows()
# imDisplay('tesla.jpg', 2)
# img2 = transformRGB2YIQ(imgArray2)
# histogramEqualize(imgArray)

# quantizeImage(imgArray, 5, 5)
