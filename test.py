import numpy as np
import cv2
import sys
import os
from time import time
import argparse

import KCFTracker
from tools.kalman_filter import KalmanFilter

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

interval = 1
duration = 0.01

def to_xyah(box):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = box.astype(np.float32).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

def to_tlwh(mean):
    """Get current position in bounding box format `(top left x, top left y,
    width, height)`.

    Returns
    -------
    ndarray
        The bounding box.

    """
    ret = mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if(abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if(w > 0):
            ix, iy = x - w / 2, y - h / 2
            initTracking = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, help='Path to video', required=True)
    args = parser.parse_args()

    assert os.path.exists(args.video), "Video doesn't exist!"
    cap = cv2.VideoCapture(args.video)
    interval = 30

    tracker = KCFTracker.KCFTracker(multiScale=False)
    kf = KalmanFilter()
    mean, cov = None, None

    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', draw_boundingbox)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        frameDraw = frame.copy()

        if(selectingObject):
            cv2.rectangle(frameDraw, (ix, iy), (cx, cy), (0, 255, 255), 1)
        elif(initTracking):
            cv2.rectangle(frameDraw, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
            print([ix, iy, w, h])
            tracker.init([ix, iy, w, h], frame)
            mean, cov = kf.initiate(to_xyah(np.asarray([ix, iy, w, h])))

            initTracking = False
            onTracking = True
        elif(onTracking):
            t0 = time()
            mean, cov = kf.predict(mean, cov)
            kcfUpdated, boundingbox = tracker.update(frame, to_tlwh(mean))
            if kcfUpdated:
                mean, cov = kf.update(mean, cov, to_xyah(boundingbox))
            else:
                print("Used KF")
                # boundingbox = to_tlwh(mean)
            t1 = time()

            boundingbox = list(map(int, boundingbox))
            cv2.rectangle(frameDraw, (boundingbox[0], boundingbox[1]), (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)

            duration = 0.8 * duration + 0.2 * (t1 - t0)
            #duration = t1-t0
            cv2.putText(frameDraw, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('tracking', frameDraw)
        c = cv2.waitKey(interval) & 0xFF
        if c == 27 or c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
