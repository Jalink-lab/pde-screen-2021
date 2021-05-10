"""
Support functions for PDE_Analysis
"""
import re
from pathlib import Path
import numpy as np
from tifffile import TiffFile
from tifffile import imwrite
from cellpose import models
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def save_mean_in_time_of_tiffs(inpath, outpath, redo=False, default_frameinterval=5.0):
    """
    Open all .tiff files and save the mean of the data in time
    :param outpath: will output mean .tif files
    :param inpath: will search the path for .tif files
    :param redo: redo the analysis
    :param default_frameinterval: the frametime to set if not present
    Note: the inpath contains multi-image-tif files, the outpath contains
    a single-image tif file with _mean added to the filename.
    """
    outpath.mkdir(parents=True, exist_ok=True)
    for file in inpath.iterdir():
        if file.suffix == '.tif':
            tif_file = TiffFile(file)
            new_file = Path(outpath, file.name[:-4] + "_mean.tif")
            if new_file.exists() and not redo:
                continue
            # Copy the metadata
            metadata = tif_file.imagej_metadata
            clean_metadata = {}
            if metadata is not None:
                for key in metadata:
                    if type(metadata[key]) in {str, int, bool, float}:
                        clean_metadata[key] = metadata[key]
            if 'finterval' not in clean_metadata.keys():
                print(f'WARNING: frame interval not in {file} using default value of {default_frameinterval} seconds')
                clean_metadata['finterval'] = default_frameinterval

            img = tif_file.asarray()
            img = img.mean(axis=0)  # the images are static, we average over all frames
            imwrite(new_file, img.astype(np.float32), metadata=clean_metadata)


def segment_cells(tiffilesin, tiffilesout, channels, gpu=True, batch_size=8, redo=False):
    """
    :param tiffilesout: list of tif files to write the labelmaps to
    :param tiffilesin: list of intensity tif files to read
    :param channels: used by cellpose
    :param gpu: use GPU
    :param batch_size: for cellpose number of 224x224 patches to run simultaneously on the GPU
    :param redo: redo the result
    :return: mask
    """
    imgs = []
    meta_data = []
    new_tiffilesout = []
    # read all .tif files, check if outputfile already exists, if not, add to list.
    for ct, file in enumerate(tiffilesin):
        if tiffilesout[ct].exists() and not redo:
            continue
        tif_file = TiffFile(file)
        img = tif_file.asarray()
        md = tif_file.shaped_metadata[0]
        md.pop('shape')  # Shape shows up in the meta-data, but is not needed.
        meta_data.append(md)
        imgs.append(img)
        new_tiffilesout.append(tiffilesout[ct])
    # run automatic segmentation
    model = models.Cellpose(gpu=gpu, model_type='cyto')
    labelmaps, flows, styles, diams = model.eval(imgs, diameter=20, channels=channels,
                                                 cellprob_threshold=0.5, do_3D=False, batch_size=batch_size)
    for i, new_file in enumerate(new_tiffilesout):
        new_file.parent.mkdir(parents=True, exist_ok=True)
        imwrite(new_file, labelmaps[i].astype(np.uint16), metadata=meta_data[i])


def get_lifetimes(componentdata, labelmap, ertr, tau=None, remove_last_frame=True):
    """
    Extracts lifetimes from the tiff files
    """
    errors = [ertr.newerror("ROI does not excist"),
              ertr.newerror("ROI smaller than min_pix"),
              ertr.newerror("ROI mean intensity smaller than min_int"),
              ertr.newerror("ROI intensitytrace has one or more zeros")]
    if tau is None:
        tau = [0.6, 3.4]
    if remove_last_frame:
        componentdata = componentdata[:-1]
    n_roi = np.max(labelmap)
    frames = componentdata.shape[0]
    intensitytraces = np.empty((n_roi, frames))
    lifetimetraces = np.empty((n_roi, frames))
    roi_size = np.empty(n_roi)  # make a new array to store ROI sizes in pixels
    for roi in range(n_roi):
        if ertr.haserror(roi):
            continue
        mask = labelmap == (roi + 1)
        roi_size[roi] = mask.sum()
        intensity = componentdata * mask
        # sum in x and y
        intensity = intensity.sum(axis=3)
        intensity = intensity.sum(axis=2)
        # sum both channels
        intensitytrace = intensity.astype(np.float64).sum(axis=1)
        # set an error for the trace that has a 0 intensity
        if (intensitytrace == 0).any():
            ertr.seterror(roi, errors[3])
        # Make the lifetimetrace NaN where intensity is 0
        intensitytrace[intensitytrace == 0] = np.inf
        lifetimetrace = (intensity[:, 0] * tau[0] + intensity[:, 1] * tau[1]) / intensitytrace
        intensitytrace = intensitytrace / mask.sum()  # store the mean per pixel
        intensitytraces[roi, :] = intensitytrace
        lifetimetraces[roi, :] = lifetimetrace

    return intensitytraces, lifetimetraces, ertr, roi_size


def fit_lifetime_traces(lifetimetraces, frameinterval, ertr, intensitytraces, roi_size, stabilityrange=0.2,
                        maxerror=0.03, start_diff=0.3, sigma_measurement=0.025, forskendpoint=False, debug=False):
    """
    Fit the lifetime traces with a sigmoidal function
    :param start_diff: allowed difference between the start of the fit and the start of the data
    :param debug: if true each graph is plotted
    :param maxerror: maximum error on the decayrate
    :param forskendpoint: True if trace ends with the rise caused by foskolin
    :param stabilityrange: allowed deviation from the mean in the first four frames
    :param ertr: error tracer, an instance of ErrorTracer that holds the possible error for each ROI
    :param lifetimetraces: input data
    :param frameinterval:
    :param sigma_measurement: in pico seconds
    :param roi_size: array of roi sizes in pixels
    :param intensitytraces: traces of the intensity of the roi
    """
    errors = [ertr.newerror("Trace starts outside of the stabilityrange"),
              ertr.newerror("Trace has NaNs"),
              ertr.newerror("Trace gave a runtime error in the fit"),
              ertr.newerror("Trace error on the rate is bigger than maxerror"),
              ertr.newerror("Fit does not start around the start of the data")]
    lifetimetraces_selected = lifetimetraces[ertr.errorfree()]
    nroi = lifetimetraces.shape[0]
    ntime = lifetimetraces.shape[1]

    # find the fit-range
    fitrange = np.zeros(shape=2, dtype=int)
    mean_trace = lifetimetraces_selected.mean(axis=0)

    d_mean_trace = np.diff(mean_trace)
    fitrange[0] = mean_trace[0:int(150 / frameinterval)].argmax()  # maximum in the first 150 seconds
    fitrange[1] = ntime
    if forskendpoint:
        # 3 frames before maximum increase after peak.
        fitrange[1] = fitrange[0] + d_mean_trace[fitrange[0]:].argmax() - 1
    xdat_org = np.arange(0, ntime * frameinterval, frameinterval)
    mean_start = mean_trace[1:5].mean()
    # required to filter tracks with abberant starting point (first point on FALCON sometimes fishy)
    columns = ['start(ns)', 'range(ns)', 'breakdown_time(s)', 'midpoint(s)', 'start tau(ns)',
               'e_start', 'e_range', 'e_breakdown_time', 'e_midpoint', 'RMSD', 'MAPE', 'error', 'condition',
               'frameinterval(s)', 'well_ID']
    fitresults = pd.DataFrame(np.nan, index=range(nroi), columns=columns)
    fitresults.fillna(0)
    for trace_id in range(nroi):
        # olga adds ROI mean instensty value, ROI size in pixels and ROI number to each line in the dataframe
        roi_int = intensitytraces[trace_id]
        # takes the mean intensity of the ROI in the first 10 frames excluding the very first
        mean_initint = roi_int[1:11].mean()
        fitresults.at[trace_id, 'ROI_nr'] = trace_id + 1
        fitresults.at[trace_id, 'Mean_init_int'] = mean_initint
        fitresults.at[trace_id, 'ROI_pix'] = roi_size[trace_id]
        # error 8 is when there is NaN's in the trace. be aware.
        if ertr.haserror(trace_id) & (ertr.geterror(trace_id) < 8):
            continue
        ydat = lifetimetraces[trace_id]
        # throw away the nan's in x and y
        xdat = xdat_org[ydat != np.nan]
        ydat = ydat[ydat != np.nan]
        start_tau = ydat[1:5].mean()  # skip first and take next four values
        if (ydat[1:5] < (mean_start - stabilityrange)).any() or (ydat[1:5] > (mean_start + stabilityrange)).any():
            ertr.seterror(trace_id, errors[0])
        p0 = np.array([3.2, -1, 0.05, xdat.mean()])  # start, range, decay-rate, half-time
        bounds = ([0, -3.4, 0, 0], [3.4, 0, 1, np.max(xdat)])  # lower and upper fitting bounds;
        xdat_selected = xdat[fitrange[0]:fitrange[1]]
        ydat_selected = ydat[fitrange[0]:fitrange[1]]
        if np.isnan(ydat_selected).any():
            ertr.seterror(trace_id, errors[1])
        if debug:
            plt.figure(1)
            plt.clf()
            plt.plot(xdat, ydat)
            plt.plot(xdat_selected, ydat_selected)
            plt.title(f"ROI : {trace_id} ; error : {ertr.geterror(trace_id)}")
            plt.show()
            plt.pause(1)
        try:
            popt, pcov = curve_fit(_fit_function, xdat_selected, ydat_selected, p0=p0, bounds=bounds)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError:
            ertr.seterror(trace_id, errors[2])
            fitresults.at[trace_id, 'error'] = ertr.geterror(trace_id)
            continue  # if there is a RuntimeError there is nothing to put into the fitresults
        if perr[2] > maxerror:
            ertr.seterror(trace_id, errors[3])
        if np.abs(popt[0] - ydat_selected[0]) > start_diff:
            ertr.seterror(trace_id, errors[4])
        ydat_fit_selected = _fit_function(xdat_selected, popt[0], popt[1], popt[2], popt[3])
        chi_sq = (((ydat_selected - ydat_fit_selected) / sigma_measurement) ** 2).sum()
        dof = fitrange[1] - fitrange[0] - 4
        rmsd = np.sqrt(((ydat_selected - ydat_fit_selected) ** 2).mean())
        mape = 100 * (np.abs(ydat_selected - ydat_fit_selected) / np.abs(ydat_selected)).mean()
        fitresults.at[trace_id, 'start(ns)'] = popt[0]
        fitresults.at[trace_id, 'range(ns)'] = popt[1]
        fitresults.at[trace_id, 'breakdown_time(s)'] = (4 / popt[2])
        fitresults.at[trace_id, 'midpoint(s)'] = popt[3]
        fitresults.at[trace_id, 'start tau(ns)'] = start_tau
        fitresults.at[trace_id, 'e_start'] = perr[0]
        fitresults.at[trace_id, 'e_range'] = perr[1]
        # d/dx 4/x = -4/x^2 ;
        fitresults.at[trace_id, 'e_breakdown_time'] = np.sqrt((-4 / (popt[2] ** 2)) ** 2 * perr[2] ** 2)
        fitresults.at[trace_id, 'e_midpoint'] = perr[3]
        fitresults.at[trace_id, 'chi_sq'] = chi_sq
        fitresults.at[trace_id, 'DOF'] = dof
        fitresults.at[trace_id, 'RMSD'] = rmsd
        fitresults.at[trace_id, 'MAPE'] = mape
        fitresults.at[trace_id, 'error'] = ertr.geterror(trace_id)
        fitresults.at[trace_id, 'ROI_nr'] = trace_id + 1
        if debug:
            plt.figure(1)
            plt.clf()
            plt.plot(xdat, ydat)
            plt.plot(xdat_selected, ydat_selected)
            plt.plot(xdat, _fit_function(xdat, popt[0], popt[1], popt[2], popt[3]))
            plt.title(f"ROI : {trace_id} ; error : {ertr.geterror(trace_id)}")
            plt.show()
            plt.pause(1)

    return fitresults, ertr


def _fit_function(x, a, b, c, d):
    return a + b / (1 + np.exp(-c * (x - d)))


def locatewell(find_well, pth):
    """
    Will attempt to find the well in the folder. If more than one well is found it will warn the user.
    :param find_well:
    :param pth:
    :return:
    """
    foundfiles = []
    for file in pth.iterdir():
        well = re.search('\D\d{1,2}', file.name)  # seach for a letter followed by one or two numbers
        if well:
            well = well.group(0)
            if well == find_well:
                foundfiles.append(file)
    if len(foundfiles) == 0:
        print(f"Did not find {find_well} in {pth}")
        return foundfiles
    elif len(foundfiles) > 1:
        print(f"Warning, found {len(foundfiles)} files:")
        for cf in foundfiles:
            print(cf.name)
        print(f"Taking {foundfiles[0].name}")
        return foundfiles[0]
    else:
        return foundfiles[0]


class ErrorTracer:
    """
    A class that stores the errors
    Example:
        ertr = ErrorTracer(100)  # Creates a tracer for 100 entries
        myError = ertr.newerror("It ate my homework!")  # create a new error and store a reference to it
        mySecondError = ertr.newerror("It just failed!")  # create a new error and store a reference to it
        ertr.seterror(20, myError)  # sets the error to entry 20 to be myError
        ertr.haserror(20)  # will return true
        ertr.whaterror(20)  # will return ["It ate my homework!"]
    """

    def __init__(self, totalnr):
        self.ids = np.zeros(totalnr, dtype='uint32')
        self.errors = []
        self.eN = -1

    def newerror(self, name):
        """
        Create a new error type
        :param name:
        :return:
        """
        self.errors.append(name)
        self.eN += 1
        return self.eN

    def seterror(self, idx, errornumber):
        """
        Set error on one of the entries
        :param idx:
        :param errornumber:
        :return:
        """
        if errornumber > len(self.errors):
            print("Error name not yet known. Please define first with ErrorTracer.newerror.")
            return
        self.ids[idx] += 2 ** errornumber

    def geterror(self, idx):
        """
        get the numerical error
        :param idx:
        :return:
        """
        return self.ids[idx]

    def savetxt(self, path):
        """
        save the error list
        :param path:
        """
        np.savetxt(path, self.ids, fmt='%i', delimiter="\t")

    def haserror(self, idx):
        """
        Does the entrie have an error
        :param idx:
        :return:
        """
        return self.ids[idx] > 0

    def errorfree(self):
        """
        Check if the set is errorfree
        :return:
        """
        return self.ids == 0

    def whaterror(self, idx):
        """
        return all errors for the entry
        :param idx:
        :return:
        """
        val = self.ids[idx]
        errors = []
        for i in range(self.eN + 1):
            if val & (1 << i):
                errors.append(self.errors[i])
        return errors
