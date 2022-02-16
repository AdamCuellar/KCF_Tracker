import cv2
import numpy as np
import fhog
import faulthandler

faulthandler.enable()


def xywh2xyxy(roi):
    xyxy = np.zeros_like(roi)
    xyxy[0] = roi[0] - roi[2] // 2
    xyxy[1] = roi[1] - roi[3] // 2
    xyxy[2] = roi[0] + roi[2]
    xyxy[3] = roi[1] + roi[3]
    return xyxy


def fftd(img, backwards=False):
    if backwards:
        return np.real(np.fft.ifft2(img, axes=(0, 1)))
    else:
        return np.fft.fft2(img, axes=(0, 1))

def fhogFeatures(img, binSize=8, nOrients=9, clip=.2, softBin=-1, useHog=2):
    nChns = nOrients if useHog == 0 else (nOrients * 4 if useHog == 1 else nOrients*3+5)
    img = img / 255
    h, w = img.shape[:2]
    M = np.zeros((h, w), dtype='float32')
    O = np.zeros((h, w), dtype='float32')
    H = np.zeros([img.shape[0] // binSize, img.shape[1] // binSize, nChns], dtype='float32')
    fhog.gradientMag(img.astype(np.float32), M, O, 0, 1)
    fhog.gradientHist(M, O, H, binSize, nOrients, softBin, useHog, clip)
    return H


def grayFeatures(img):
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = img - img.mean()
    return img


class KCFTracker:
    def __init__(self, featureType="hog", multiScale=False):
        self.featureType = featureType
        self.multiScale = multiScale

        # initialize lambda, padding, and output sigma factor
        self.lambd = 1e-4
        self.padding = 1.5
        self.numClusters = 15

        # initialize some variables that we may modify below
        self.interpFactor = 0.075
        self.outSigmaFactor = 0.1
        self.kernelSigma = 0.2
        self.cellSize = 1
        self.scaleStep = 1
        self.scaleWeight = 1
        self.hog = "hog" in featureType
        self.hogOrientations = 9

        if self.hog:
            self.interpFactor = 0.02
            self.kernelSigma = 0.5
            self.cellSize = 4

        if self.multiScale:
            self.scaleStep = 1.05
            self.scaleWeight = 0.95

        self._tempF = None
        self._tempShape = np.zeros(2)
        self._roi = None
        self._gaussF = None
        self._alphaF = None
        self._hannWindow = None
        self._outputSigma = 0

    def init(self, roi, image):
        # roi is expected to be a numpy array of xmin, ymin, width, height
        self._roi = np.asarray(roi).astype(np.float32)
        self._tempShape[:] = self._roi[2:4] * (1 + self.padding)
        self._tempShape = self._tempShape.astype(np.int32)
        self._outputSigma = np.sqrt(np.prod(self._tempShape)) * self.outSigmaFactor / self.cellSize
        self._gaussF = self.createGaussianPeak(self._tempShape // self.cellSize)
        self._hannWindow = cv2.createHanningWindow(self._tempShape // self.cellSize, cv2.CV_32F)
        self._tempF = self.getFeatures(image)
        self._alphaF = np.zeros((self._tempF.shape[0], self._tempF.shape[1]), dtype=np.float32)
        self.train(self._tempF, 1.0)

    def getFeatures(self, image, scaleAdjust=1.0):
        extractedRoi = np.zeros((4))
        cx = self._roi[0] + self._roi[2] / 2
        cy = self._roi[1] + self._roi[3] / 2

        # adjust extracted roi
        extractedRoi[2:] = self._tempShape * scaleAdjust
        extractedRoi[:2] = np.asarray([cx, cy])
        extractedRoi = extractedRoi.astype(np.int32)

        # make sure the roi is in bounds, if anything is out of bounds replicate the values
        xs = np.floor(extractedRoi[0]) + np.arange(0, extractedRoi[2]) - extractedRoi[2] // 2
        xs = xs.astype(np.int32)
        ys = np.floor(extractedRoi[1]) + np.arange(0, extractedRoi[3]) - extractedRoi[3] // 2
        ys = ys.astype(np.int32)
        xs[xs < 1] = 1
        ys[ys < 1] = 1
        xs[xs > image.shape[1]] = image.shape[1] - 1
        ys[ys > image.shape[0]] = image.shape[0] - 1
        window = image[ys, ...][:, xs, :]

        if window.shape[0] != self._tempShape[1] or window.shape[1] != self._tempShape[0]:
            window = cv2.resize(window, self._tempShape)

        if self.hog:
            features = fhogFeatures(window, binSize=self.cellSize, nOrients=self.hogOrientations)
            features = features[:,:,:-1] # remove all zeros channel
        else:
            features = grayFeatures(window)

        features *= self._hannWindow[:, :, np.newaxis]
        return fftd(features)

    def createGaussianPeak(self, shape):
        w, h = shape
        halfH, halfW = h // 2, w // 2
        y, x = np.mgrid[-halfH:h-halfH, -halfW:w-halfW]
        res = np.exp(-0.5 / (self._outputSigma ** 2) * (y**2 + x**2))
        res = np.roll(res, -1 * (np.asarray(res.shape) // 2), axis=(0, 1))
        assert res[0, 0] == 1, "Make sure Gaussian Response is in top-left"
        return fftd(res)

    def train(self, xF, train_interp_factor):
        k = self.gaussianCorrelation(xF, xF)
        alphaF = self._gaussF / (k + self.lambd)

        self._tempF = (1 - train_interp_factor) * self._tempF + train_interp_factor * xF
        self._alphaF = (1 - train_interp_factor) * self._alphaF + train_interp_factor * alphaF

    def gaussianCorrelation(self, x, y):
        N = x.shape[0] * x.shape[1]
        xx = x.flatten().conj().T @ x.flatten() / N
        yy = y.flatten().conj().T @ y.flatten() / N

        xyf = x * np.conj(y)
        xy = np.sum(fftd(xyf, True), axis=-1)
        kf = fftd(np.exp(-1 / (self.kernelSigma ** 2) * np.maximum(0, (xx + yy - 2 * xy) / x.size)))
        return kf

    def subPixelPeak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)

    def detect(self, template, features):
        k = self.gaussianCorrelation(features, template)
        res = fftd(self._alphaF * k, True)
        maxRow, maxCol = np.unravel_index(np.argmax(res, axis=None), res.shape)
        rowDelta, colDelta = maxRow, maxCol

        if maxRow > 0 and maxRow < res.shape[1] - 1:
            rowDelta = maxRow + self.subPixelPeak(res[maxRow, maxCol-1], res.max(), res[maxRow, maxCol+1])
        if maxCol > 0 and maxCol < res.shape[0] - 1:
            colDelta = maxCol + self.subPixelPeak(res[maxRow-1, maxCol], res.max(), res[maxRow+1, maxCol])

        if rowDelta > features.shape[0] / 2:
            rowDelta -= features.shape[0]
        if colDelta > features.shape[1] / 2:
            colDelta -= features.shape[1]

        return (rowDelta, colDelta), res.max()

    def update(self, image, updatedRoi=None):
        if updatedRoi:
            self._roi = np.asarray(updatedRoi).astype(np.float32)

        if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 1
        if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[3] + 1
        if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 2
        if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        loc, peak_value = self.detect(self._tempF, self.getFeatures(image, 1.0))

        if self.scaleStep != 1:
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.detect(self._tempF,
                                                    self.getFeatures(image, 1.0 / self.scaleStep))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.detect(self._tempF, self.getFeatures(image, self.scaleStep))

            if self.scaleWeight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2:
                loc = new_loc1
                self._roi[2] /= self.scaleStep
                self._roi[3] /= self.scaleStep
            elif self.scaleWeight * new_peak_value2 > peak_value:
                loc = new_loc2
                self._roi[2] *= self.scaleStep
                self._roi[3] *= self.scaleStep

        self._roi[0] = (cx + loc[1] * self.cellSize) - (self._roi[2] / 2.0)
        self._roi[1] = (cy + loc[0] * self.cellSize) - (self._roi[3] / 2.0)

        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[3] + 2

        assert (self._roi[2] > 0 and self._roi[3] > 0)

        x = self.getFeatures(image, 1.0)
        self.train(x, self.interpFactor)

        return self._roi