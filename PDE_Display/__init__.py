"""
PDE_Display contains helper functions for displaying the result of PDE_Analysis
"""
import json
import sys
from pathlib import Path

import numpy as np
from tifffile import TiffFile


class Config:
    """
    Config object with some simple read/write operations.
    Example:
        from AutomatedCellAnalysis.config import Config
        cfg = Config()
        myPath = cfg.readorwrite('myPath','c:\\')
        // If myPath exists in the config file it is returned, if not it is written
        // to the config file and 'c:\\' is returned
        myPath = cfg.read('myPath', default='c:\\')
        // If myPath exists in the config file it is returned, if not 'c:\\' is returned
    """

    def __init__(self, cfg_path=None):
        if cfg_path is None:
            cfg_path = Path(Path.home(), 'jalink_pde_config.ini')
        self.cfg_file = cfg_path
        if not self.cfg_file.exists():
            self.clear()

    def read(self, name, **kwargs):
        """
        Read the value corresponding to a name
        """
        z = json.loads(self.cfg_file.read_text())
        return z.get(name, kwargs.get('default', None))

    def readorwrite(self, name, default):
        """
        Tries to read the config value.
        If it cannot find it, it returns the default value and adds that value to the config file.
        """
        res = self.read(name, default=None)
        if res is None:
            self.write(name, default)
            return default
        else:
            return res

    def clearname(self, name):
        """
        Clear a name from the config file
        """
        z = json.loads(self.cfg_file.read_text())
        if name in z.keys():
            value = z.pop(name)
            self.cfg_file.write_text(json.dumps(z))
            return value
        else:
            return None

    def list(self):
        """
        List all names in the config file
        """
        z = json.loads(self.cfg_file.read_text())
        return z.keys()

    def write(self, name, value):
        """
        Write a name-value pair to a config file
        """
        if self.cfg_file.exists():
            z = json.loads(self.cfg_file.read_text())
            z.update({name: value})
        else:
            z = {name: value}
        self.cfg_file.write_text(json.dumps(z))

    def clear(self):
        """
        Clear the entire config file
        """
        if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
            self.cfg_file.unlink(missing_ok=True)
        else:
            if self.cfg_file.exists():
                self.cfg_file.unlink()
        self.cfg_file.write_text('{}')


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
