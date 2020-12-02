# PatchExtractor
This project is part of a pipeline developed for the Venice Time Machine project.
The purpose of this repo is to provide the tools required to extract patches from the Venetian cadastre images.

This repository is the second step in a pipeline to process the Venetian Sommarioni cadastre. 
For the full pipeline description please read the main [repository](https://github.com/Jmion/VeniceTimeMachineSommarioniHTR.git).
A full description of the pipeline is also available on our [wiki](http://fdh.epfl.ch/index.php/Deciphering_Venetian_handwriting).

## 🚀 Installation 

To install and run this script please use the ```environment.yml```.
If you need a quick reminder on how to do so here are the commands:
```
conda env create -f environment.yml
conda activate patchExtract
```

One that is done do ahead and run ```python extractPatches.py --help``` to see the help page and parameters available.

Please find below the help page:

```
usage: extractPatches.py [-h] [--offset_above offsetAbove] [--offset_below offsetBelow] [--disable_progress_bar DISABLE_PROGRESS_BAR] [--page page]
                         imgDir xmlDir patchDir dfDir

This script is to be ruin after running p2pala to extract the baselines out of handwritten images. It uses will analyse the coordinates produced to create a patch
around the baseline. These patches will be saved to a directory to allow for processing with htr methods.

positional arguments:
  imgDir                directory containing the source images.  The images must end in *.jpg
  xmlDir                directory containing the xml.
  patchDir              directory where the patches will be saved. If non existant will be created
  dfDir                 Locations where the dataframes contains x_min, y_min, x_max, y_max, filename, pagename; will be saved as a csv. If the directory does not exist
                        it will be created.

optional arguments:
  -h, --help            show this help message and exit
  --offset_above offsetAbove, -oa offsetAbove
                        Offset above baseline to take. Larger values will increase the room above the patch.
  --offset_below offsetBelow, -ob offsetBelow
                        Offset below baseline to take. Larger values will increase the room below the patch.
  --disable_progress_bar DISABLE_PROGRESS_BAR
                        dissables progress bar
  --page page           page name, jpg filename of the page to run the extraction on. Usefull for debuging if there is a crash on a specific page.

```

If you want to run it as an executable file please modify the shabang in the extractPatches.py to point to your conda environment that you just installed.
