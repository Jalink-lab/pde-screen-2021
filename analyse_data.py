"""
The Experimental Data used in this screen has been collected on 5 seprate days
 *  06-02-2020 (knockdown of PDE 3A and PDE 10 half with caged and half with IsoP/Prop/Forsk)
 *  07-01-2020 (extra IBMX and only cells with IsoP Prop and Forsk)
 *  05-12-2019 (caged cAMP)
 *  07-11-2019 (with IsoP/Prop/Forsk)
 *  02-05-2019 (with IsoP/Prop/Forsk)

The data is stored in a public Folder called Screening_Data_Notanalyzed
The results will be stored in a new folder called Screening_Data_Analyzed in a folder tree that
mirrors that of the input data.
The data cosists of:
    * intensity timelaps tiff files for each well
    * lifetime tiff files containing two components of the donor lifetime resulting from LASX fit
    * 96-well plate layout of that particular experiment
This routine analyse_data calls functions from PDE_Analysis, in order:
    - save_mean_in_time_of_tiffs which generates average intensity images used for segmentation
    - segment_cells which uses CellPose for deep-learning segmentation
    - get_lifetimes
    - fit_lifetime_traces
"""
from pathlib import Path
import pandas as pd
import numpy as np
from tifffile import TiffFile
from PDE_Analysis import save_mean_in_time_of_tiffs
from PDE_Analysis import segment_cells
from PDE_Analysis import locatewell
from PDE_Analysis import ErrorTracer
from PDE_Analysis import get_lifetimes
from PDE_Analysis import fit_lifetime_traces

# SET THE PATH TO THE RAW DATA HERE:
ROOT_PATH = Path('K:\\', 'SurfDrive', 'Shared', 'Screening_Data_Notanalyzed')
# SET THE PATH TO WRITE DATA TO HERE:
WRITE_PATH = Path('D:\\', 'Temp', 'Screening_Result')
# IF SET TO TRUE THE PREVIOUS ANALYSIS IS OVERWRITTEN
REDO = False

# set paths to input and output of data and if they have foskolin as endpoint callibration
forskendpoints = [True, True, False, True, True, False]

paths = [Path('2019', '05', '02', 'chemical'),
         Path('2019', '11', '07', 'chemical'),
         Path('2019', '12', '05', 'caged'),
         Path('2020', '01', '07', 'chemical'),
         Path('2020', '02', '06', 'chemical'),
         Path('2020', '02', '06', 'caged')]
inpaths = []
outpaths = []
for path in paths:
    inpaths.append(Path(ROOT_PATH, path))
    outpaths.append(Path(WRITE_PATH, path))

# For each experiment (timelapse) generate the mean of all the timepoints (tiffiles) in the folder
for ct, inpath in enumerate(inpaths):
    print(inpath)
    outpath = outpaths[ct]
    save_mean_in_time_of_tiffs(Path(inpath, 'intensity_data'), Path(outpath, 'intensity_data'), redo=REDO)

# group files per folder and run the segment_cells (this is by far the most time consuming process)
for outpath in outpaths:
    infiles = []
    outfiles = []
    channels = []
    curr_path = Path(outpath, 'intensity_data')  # this is the folder with the _mean intensity data
    lbl_path = Path(outpath, 'labelmap_data')  # this is the folder with the _mean intensity data
    for file in curr_path.iterdir():
        if file.name[-9:] == '_mean.tif':
            infiles.append(file)
            outfiles.append(Path(lbl_path, file.name[:-4] + "_labelmap.tif"))
            channels.append([1, 2])  # first channel is cytosol, second channel is dapi
    segment_cells(infiles, outfiles, channels[0], redo=REDO, gpu=True, batch_size=4)

# get dataframe with fit data from segmentation and 2-component fit-data
for ct, outpath in enumerate(outpaths):
    inpath = inpaths[ct]
    layout_file = Path(inpath, 'PlateLayout.txt')
    labelmap_path = Path(outpath, 'labelmap_data')
    components_path = Path(inpath, 'two_comp_fit')
    results_file = Path(outpath, 'results', "all_results.csv")
    forskendpoint = forskendpoints[ct]
    results_file.parent.mkdir(parents=True, exist_ok=True)
    # read the labelmap file
    layout = []
    for txt_line in open(layout_file):
        txt_line = txt_line.rstrip()  # remove all spaces and returns at the end of the line
        layout.append(txt_line.split(", "))
    all_data = pd.DataFrame()
    for txt_line in layout:  # each condition
        for i in range(1, len(txt_line)):  # each CONDITION, so, the two duplicate wells together
            fitfile = Path(results_file.parent, txt_line[i] + "_fit.csv")
            if fitfile.exists() and not REDO:
                fitvalues = pd.read_csv(fitfile)  # need it for creating all_data
            else:
                well = txt_line[i]
                print(f"working on well {well} with condition {txt_line[0]}")
                componentfile = locatewell(well, components_path)
                labelmapfile = locatewell(well, labelmap_path)
                componenttiff = TiffFile(componentfile)
                labelmaptiff = TiffFile(labelmapfile)
                if 'finterval' not in labelmaptiff.shaped_metadata[0]:
                    print(f"ERROR: Frame interval not found for {labelmapfile}")
                    break
                else:
                    frameinterval = labelmaptiff.shaped_metadata[0]['finterval']
                labelmap = labelmaptiff.asarray()
                ertr = ErrorTracer(np.max(labelmap))  # trace errors for each ROI in the labelmap
                intensitytraces, lifetimetraces, ertr, roi_size = get_lifetimes(componenttiff.asarray(),
                                                                                labelmap[::2, ::2], ertr)
                np.savetxt(Path(results_file.parent, txt_line[i] + "_tau.csv"), lifetimetraces, delimiter="\t")
                np.savetxt(Path(results_file.parent, txt_line[i] + "_int.csv"), intensitytraces, delimiter="\t")
                [fitvalues, ertr] = fit_lifetime_traces(lifetimetraces, frameinterval, ertr, intensitytraces, roi_size,
                                                        forskendpoint=forskendpoint)
                fitvalues['condition'] = txt_line[0]
                fitvalues['frameinterval(s)'] = frameinterval
                fitvalues['well_ID'] = txt_line[i]
                fitvalues = fitvalues.replace([np.inf, -np.inf], np.nan)
                ertr.savetxt(Path(results_file.parent, txt_line[i] + "_errors.csv"))
                fitvalues.to_csv(fitfile)
            all_data = pd.concat([all_data, fitvalues])
    all_data.to_csv(results_file)
