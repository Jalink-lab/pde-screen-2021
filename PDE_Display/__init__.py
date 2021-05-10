"""
PDE_Display contains helper functions for displaying the result of PDE_Analysis
"""
from tifffile import TiffFile
import numpy as np


def get_cell_image(labelmap_pth, meanimg_pth, roinumber):
    """
    get the cell highlighted in an image
    """
    labelmap = TiffFile(labelmap_pth)
    labelmap_img = labelmap.asarray()
    meanimg = TiffFile(meanimg_pth)
    meanimg_img = meanimg.asarray()
    if meanimg_img.ndim == 3:
        meanimg_img = meanimg_img[0, :, :]
    meanimg_img = meanimg_img.astype('float')
    meanimg_img = meanimg_img - np.min(meanimg_img)
    meanimg_img = meanimg_img / np.max(meanimg_img)
    labelmap_img_selected = labelmap_img == roinumber
    npix = labelmap_img_selected.sum()
    if npix == 0:
        print("ROI number not found")
        return
    meanimg_img_color = np.zeros([meanimg_img.shape[0], meanimg_img.shape[1], 3], 'float')
    meanimg_img_highlighted = meanimg_img.copy()
    meanimg_img_highlighted[labelmap_img_selected] += 0.25
    meanimg_img_color[:, :, 0] = meanimg_img  # Red
    meanimg_img_color[:, :, 1] = meanimg_img_highlighted  # Green
    meanimg_img_color[:, :, 2] = meanimg_img  # Blue

    return meanimg_img_color
