import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

# Reading an image into a given representation
if __name__ == '__main__':
    # 3.1: display a given image file in a given representation
    def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
        filepath = './pics/' + filename
        print(filepath)
        if representation == 1:
            channel = 0  # zero for greyscale
        elif representation == 2:
            channel = 1  # one for RGB
        else:
            raise ValueError('only 1 or 2 are possible inputs for representation, '
                             'to output Greyscale or RGB images respectively')
        img = cv2.imread(filepath, channel)
        if img is None:
            raise ValueError("Could not find requested file inside 'pics' folder. ")
        img = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return img


    # 3.2: displaying an image. raised errors from imReadAndConvert will be raised forward.
    def imDisplay(filename: str, representation: int):
        img = imReadAndConvert(filename, representation)
        cv2.imshow('Liad and Timor showing image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 3.3 part A: transform RGB to YIQ
    def transformRGB2YIQ(imRGB: np.ndarray) -> np.ndarray:
        mat = np.array([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
        for x in imRGB:
            for y in x:
                y[:] = mat.dot(y[:])  # linear matrix multiplication
        return imRGB

    # 3.3 part B: transform YIQ to RGB
    def transformYIQ2RGB(imYIQ: np.ndarray) -> np.ndarray:
        mat = np.array([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
        mat = np.linalg.inv(mat)  # inverse matrix

        for x in imYIQ:
            for y in x:
                y[:] = mat.dot(y[:])  # linear matrix multiplication
        return imYIQ

    # 3.4 Histogram equalization
    def histogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        global histOrig, histEq
        if len(imOrig.shape) == 2:
            greyscale = True
        elif len(imOrig.shape) == 3:
            greyscale = False
        else:
            raise ValueError('Unsupported array representation. only RGB or Greyscale images allowed')
        imEq = np.copy(imOrig)
        if greyscale:
            imEq = cv2.normalize(imEq, None, 0, 255, cv2.NORM_MINMAX)
            imEq = np.ceil(imEq)  # floating point precision correction
            imEq = imEq.astype('uint8')

            # Original Histogram:
            histOrig, bins = np.histogram(imEq.flatten(), 256, [0, 255])
            cdf = histOrig.cumsum()  # cumulative
            cdf_normalized = cdf * histOrig.max() / cdf.max()

            plt.title('Original image histogram with CDF (Liad & Timor)')
            plt.plot(cdf_normalized, color='b')
            plt.hist(imEq.flatten(), 256, [0, 255], color='r')
            plt.xlim([0, 255])
            plt.legend(('cdf - ORIGINAL', 'histogram - ORIGINAL'), loc='upper left')
            plt.show()

            # equalize operation on the copy of original image
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            imEq = cdf[imEq]

            # histogram for equalized image:
            histEq, bins = np.histogram(imEq.flatten(), 256, [0, 255])
            cdf = histEq.cumsum()  # cumulative
            cdf_normalized = cdf * histEq.max() / cdf.max()

            plt2.title('Equalized image histogram with CDF (Liad & Timor)')
            plt2.plot(cdf_normalized, color='b')
            plt2.hist(imEq.flatten(), 256, [0, 255], color='r')
            plt2.xlim([0, 255])
            plt2.legend(('cdf - EQUALIZED', 'histogram - EQUALIZED'), loc='upper right')
            plt2.show()

            # display original image
            cv2.imshow('Liad and Timor showing the ORIGINAL image', imOrig)

            # display equalized image
            cv2.imshow('Liad and Timor showing the EQUALIZED image', imEq)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return imEq, histOrig, histEq


    # Main
    try:
        img = imReadAndConvert('notOptimal.jpg', 1)
        a, b, c = histogramEqualize(img)
    except ValueError as err:
        print(err.args)
