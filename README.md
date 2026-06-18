# PySpecExtract
[![DOI](https://zenodo.org/badge/635694695.svg)](https://doi.org/10.5281/zenodo.20741124)  
Python `Tkinter` based GUI application to extract spectra from IFU cubes.
Designed to work with ANU 2.3m WiFeS reduced data cubes, but can technically work with any IFU data cube
by specifying cube dimensions and telescope-specific `.fits` headers.

Minimum python version: 3.10 (3.9 works but may break when using dark mode systems)

Requires standard scientific python libraries: `numpy` `scipy` `matplotlib` `astropy` `pandas`.  
GUI requires `Tkinter`.

-------------
# Set up
Download the `.py` files from this repository and run the circular aperture version with: 
```
python spec_GUI.py
```
or, for the PSF fitting version:
```
python spec_GUI_psf.py
```

---------------
# Circular aperture version
This version allows the user to select a circular aperture to extract a spectrum from a WiFeS data cube.
The user can select an annulus or a disjoint background region for sky subtraction. The position and radius 
of the apertures can be adjusted as needed using the GUI.

---------------
# PSF fitting version
This version allows the user to fit a Moffat (recommended) or Gaussian PSF to a point source in the IFU field of view.
The user must select a background region for sky subtraction, as the annulus option is not available for this method.
The PSF is fit in wavelength bins (default 8) across the cube, and the fitting parameters are interpolated to from a 
spline function to create a wavelength-dependent PSF model for the full cube. The initial guess for the PSF and
background region are selected using the GUI.

---------------
# Usage
- Select the directory containing the WiFeS data cubes to be processed in the first dialogue window. The program will 
search for all `.fits` files in the directory and subdirectories. Search can be limited to specific files by modifying the
`make_amalgamated_file` function in `spectra_extractor{_psf}.py`. This will generate a file called `obj_fits_list.csv`
which contains the list of file locations and unique IDs for each cube to be processed. Multiple cubes of the same
object are separated based on observed date and time. Once the file is generated, click `Continue`.  

Note: if `obj_fits_list.csv` already exists, or you have separately created it, you can skip the initial window by
replacing
```python
app = RawDirs(master)
```
at the end of the `spec_GUI{_psf}.py` file with
```python
app = MainWindow(master, "/path/to/dir/", pd.read_csv("/path/to/object_fits_list.csv"))
```
- The program will create the output directories in `out/` if they do not already exist in the working directory.
- The main GUI window will open. Navigate to the desired cube using the buttons at the bottom. The cubes are ordered
based on the order in `obj_fits_list.csv`.
- The cube will be displayed as a 2D median image collapsed along the wavelength axis. Use the option button on the
top to select either object or sky, and click a pixel on the image to set the centre of the aperture or background region.
- For PSF fitting version, the position of the object is the initial guess for the PSF centre, and only background 
aperture size can be set.
- Once ideal positions are selected, click `Save` at the bottom to save the output files to `out/`.

Note: The GUI scripts are wrappers for the main extraction functions in `spectra_extractor.py` and
`spectra_extractor_psf.py`. The extraction parameters (e.g. aperture size, background type, PSF type, number of
wavelength bins etc.) can be modified in these files directly if desired. Subsequently, the extraction functions can be
independently imported and used in other scripts/notebooks as needed by creating a `SpecExtract` object. For example:
```python
import spectra_extractor_psf as se
import matplotlib.pyplot as plt

spec_object = se.SpecExtract("ObjectName", "path/to/red.fits", "path/to/blue.fits",
                         r=2.5, sky_r=3, row=12, col=12, row_min=20, col_min=20)
spec_object.generate_spec(save_loc='path/to/save/', init=True)
spec_object.plot_spec(save_loc=f"path/to/save/").show()
```
---------------
# Output
- The extracted spectra are saved as 
[MARZ](https://skymapper.anu.edu.au/static/sm_asvo/marz/index.html#/overview) compatible `.fits` files in `out/WiFeS/`.
- The spectrum and 2D image of the cube along with aperture overlay are saved as displayed in the GUI in 
`out/spat_plots` and `out/spec_plots`. Note that any interactive elements (e.g. zoom) are not saved.
- For the PSF version, a plot of the wavelength profile of the fitting parameters are saved in `out/wave_profiles/` and the
data - model residual plots for the wavelength bins used for the PSF fitting are saved in `out/psf_fits/`.

---------------

