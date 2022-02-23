import numpy as np

def xyah2tlwh(roi):
    """
    Convert `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`. bounding box to format
    (xmin, ymin, width, height)
    """
    ret = roi.copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

def tlwh2xyah(roi):
    """
    Convert (xmin, ymin, width, height) bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = roi.copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

def xywh2xyxy(roi):
    """
    Convert (xcen, ycen, width, height) bounding box to format (xmin, ymin, xmax, ymax)
    """
    xyxy = roi.copy()
    xyxy[0] = roi[0] - roi[2] // 2
    xyxy[1] = roi[1] - roi[3] // 2
    xyxy[2] = roi[0] + roi[2]
    xyxy[3] = roi[1] + roi[3]
    return xyxy
