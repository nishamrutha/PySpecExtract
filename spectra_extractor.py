import glob
import os
from datetime import datetime

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Plotting options
rcParams['font.size'] = 11

# set default lines
rcParams['lines.linewidth'] = 1.0
rcParams['axes.linewidth'] = 0.8

# change x-axis characteristics
rcParams['xtick.top'] = True
rcParams['xtick.direction'] = 'in'
rcParams['xtick.minor.visible'] = True
rcParams['xtick.major.size'] = 4
rcParams['xtick.minor.size'] = 3
rcParams['xtick.major.width'] = 0.75
rcParams['xtick.minor.width'] = 0.25
rcParams['xtick.major.pad'] = 4
rcParams['xtick.minor.pad'] = 4

# change y-axis characteristics
rcParams['ytick.right'] = True
rcParams['ytick.direction'] = 'in'
rcParams['ytick.minor.visible'] = True
rcParams['ytick.major.size'] = 4
rcParams['ytick.minor.size'] = 3
rcParams['ytick.major.width'] = 0.75
rcParams['ytick.minor.width'] = 0.25
rcParams['ytick.major.pad'] = 4
rcParams['ytick.minor.pad'] = 4

# set default legend characteristics
rcParams['legend.fontsize'] = 7
rcParams['legend.labelspacing'] = 0.2
rcParams['legend.loc'] = 'best'
rcParams['legend.frameon'] = False

# set figure layout
rcParams['savefig.bbox'] = 'tight'

#######################################

# Constants and Catalogues
n_row, n_col = 35, 25  # WiFeS cube face dimensions, remove 3 from n_row for bad pixel rows
sep = 5650  # WiFeS red/blue separation wavelength
full_min, full_max = 4000, 7500  # WiFeS spectrum range


class SpecExtract:
    def __init__(self, obj_name, red_f, blue_f, **kwargs):
        """

        :param obj_name:
        :param red_f:
        :param blue_f:
        :param kwargs:
        """
        self.obj_name = obj_name

        print(f"Opening {obj_name}...")

        # Open red and blue FITS files
        self.red_hdu = fits.open(red_f)
        self.blue_hdu = fits.open(blue_f)

        ''' Blue '''
        self.blue = self.blue_hdu[0].data[:, 3:, :]  # Remove bad pixel rows
        self.blue_e = self.blue_hdu[1].data[:, 3:, :]
        self.blue_head = self.blue_hdu[0].header
        self.wave_blue = self.blue_head['CRVAL3'] + np.arange(len(self.blue)) * self.blue_head['CDELT3']
        self.blue_ind = np.ravel(np.argwhere((self.wave_blue > full_min) & (self.wave_blue <= sep)))
        self.blue_img = np.nanmedian(self.blue, axis=0)

        ''' Red '''
        self.red = self.red_hdu[0].data[:, 3:, :]  # Remove bad pixel rows
        self.red_e = self.red_hdu[1].data[:, 3:, :]
        self.red_head = self.red_hdu[0].header
        self.wave_red = self.red_head['CRVAL3'] + np.arange(len(self.red)) * self.red_head['CDELT3']
        self.red_ind = np.ravel(np.argwhere((self.wave_red > sep) & (self.wave_red <= full_max)))
        self.red_img = np.nanmedian(self.red, axis=0)

        # Observed date in MJD
        self.obs_mjd = dt2mjd(md.date2num(datetime.fromisoformat(self.blue_head['DATE-OBS'].split('T')[0])))

        # Aperture position
        self.row = 16
        self.col = 12

        # Sky
        self.r = 2  # aperture radius
        self.sky_aperture = 'disjoint'  # [disjoint, annular]
        self.sky_r = 2  # sky aperture radius

        # Disjoint sky position
        self.row_min = 30
        self.col_min = 12

        # Placeholders for masks and data arrays
        self.mask = None
        self.mask_min = None
        self.spec_wifes_raw = None
        self.spec_wifes_err = None
        self.wave_wifes = None

        # Set kwargs attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Close FITS
        self.red_hdu.close()
        self.blue_hdu.close()

    def make_masks(self):
        """

        """
        # Aperture mask
        x, y = np.ogrid[-self.row:n_row - self.row, -self.col:n_col - self.col]
        self.mask = x * x + y * y <= self.r ** 2

        # Aperture mask for sky subtraction
        if self.sky_aperture == 'disjoint':  # Free
            x, y = np.ogrid[-self.row_min:n_row - self.row_min, -self.col_min:n_col - self.col_min]
            self.mask_min = x * x + y * y <= self.sky_r ** 2
        elif self.sky_aperture == 'annular':  # Annular
            self.mask_min = (x * x + y * y >= self.r ** 2) & (x * x + y * y <= (self.r + self.sky_r) ** 2)
        else:
            print("Choose sky aperture from [disjoint, annular]")

    def generate_spec(self, save_loc="WiFeS_with_error/", save=True):
        """

        :param save_loc:
        :param save:
        """
        self.make_masks()  # generate aperture and sky mask

        # FITS header dictionary
        cat = {'object': self.obj_name,
               'data': "Intensity",
               'row': self.row + 3,  # de-offset bad rows when saving
               'col': self.col,
               'apt_rad': self.r,
               'OBS_MJD': self.obs_mjd,  # observed MJD
               'sky_apt': self.sky_aperture}  # sky aperture type

        # Sky aperture details in FITS header
        if self.sky_aperture == 'disjoint':
            cat['row_sky'] = self.row_min + 3  # de-offset bad rows when saving
            cat['col_sky'] = self.col_min
        elif self.sky_aperture == 'annular':
            cat['row_sky'] = self.row + 3  # de-offset bad rows when saving
            cat['col_sky'] = self.col
        else:
            print("Choose sky aperture from [disjoint, annular]")
        cat['sky_rad'] = self.sky_r

        # Average of sky
        blue_mean = np.mean(self.blue[:, self.mask_min], axis=1)
        red_mean = np.mean(self.red[:, self.mask_min], axis=1)

        # Error in sky average (variance)
        blue_sky_sem = np.mean(self.blue_e[:, self.mask_min], axis=1) / np.sqrt(
            self.blue_e[:, self.mask_min].shape[1])  # error
        red_sky_sem = np.mean(self.red_e[:, self.mask_min], axis=1) / np.sqrt(
            self.red_e[:, self.mask_min].shape[1])  # error

        # Sum of aperture after sky subtraction
        sqrt_n = np.sqrt(self.blue[:, self.mask].shape[1])
        spec_blue = np.sum(self.blue[:, self.mask] - blue_mean[:, None], axis=1)
        spec_red = np.sum(self.red[:, self.mask] - red_mean[:, None], axis=1)

        # Error in aperture after sky subtraction (standard deviation)
        err_blue = np.sqrt(np.mean(self.blue_e[:, self.mask], axis=1)
                           / np.sqrt(self.blue_e[:, self.mask].shape[1])
                           + blue_sky_sem) * sqrt_n
        err_red = np.sqrt(np.mean(self.red_e[:, self.mask], axis=1)
                          / np.sqrt(self.red_e[:, self.mask].shape[1])
                          + red_sky_sem) * sqrt_n  # subtraction error

        # Join red and blue indices
        self.spec_wifes_raw = np.concatenate((spec_blue[self.blue_ind], spec_red[self.red_ind]))
        self.spec_wifes_err = np.concatenate((err_blue[self.blue_ind], err_red[self.red_ind]))
        self.wave_wifes = np.concatenate((self.wave_blue[self.blue_ind], self.wave_red[self.red_ind]))

        # Save MARZ-friendly FITS
        if save:
            hdu = fits.PrimaryHDU(self.spec_wifes_raw)  # intensity array in primary
            hdu.header.update(cat)  # append object details dict to header
            hdu1 = fits.ImageHDU(self.wave_wifes, name='wavelength')  # wavelength array in image extension
            hdu2 = fits.ImageHDU(self.spec_wifes_err, name='int_err')  # intensity error
            fits.HDUList([hdu, hdu1, hdu2]).writeto(f'{save_loc}/{self.obj_name}.fits', overwrite=True)  # write

    def plot_spatial(self, save=False, save_loc='spat_plots/'):
        """

        :param save:
        :param save_loc:
        :return:
        """
        self.make_masks()

        # Circles to show aperture and sky on spatial image
        circle1 = plt.Circle((self.col, self.row), self.r, color='r', fill=False)
        circle2 = plt.Circle((self.col, self.row), self.r, color='r', fill=False)
        if self.sky_aperture == 'annular':
            circle3 = plt.Circle((self.col, self.row), self.r + self.sky_r, color='r', linestyle='--', fill=False)
            circle4 = plt.Circle((self.col, self.row), self.r + self.sky_r, color='r', linestyle='--', fill=False)
        else:
            circle3 = plt.Circle((self.col_min, self.row_min), self.sky_r, color='r', linestyle='--', fill=False)
            circle4 = plt.Circle((self.col_min, self.row_min), self.sky_r, color='r', linestyle='--', fill=False)

        # Mark central pixel
        rect1 = plt.Rectangle((self.col - 0.5, self.row - 0.5), 1, 1, color='r', fill=False)
        rect2 = plt.Rectangle((self.col - 0.5, self.row - 0.5), 1, 1, color='r', fill=False)

        ''' Blue '''
        fig, [axb, axr] = plt.subplots(1, 2)
        divider = make_axes_locatable(axb)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        imb = axb.imshow(self.blue_img)  # 3 bad rows at edge not shown
        fig.colorbar(imb, cax=cax, orientation='vertical')
        axb.set_title('Blue')
        axb.add_patch(circle1)
        axb.add_patch(circle3)
        axb.add_patch(rect1)

        ''' Red '''
        divider = make_axes_locatable(axr)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        imr = axr.imshow(self.red_img)  # 3 bad rows at edge not shown
        fig.colorbar(imr, cax=cax, orientation='vertical')
        axr.set_title('Red')
        axr.add_patch(circle2)
        axr.add_patch(circle4)
        axr.add_patch(rect2)
        plt.tight_layout()

        # Save
        if save:
            plt.savefig(save_loc + '/' + self.obj_name + '.pdf')

        # Return figure to plot in GUI section
        return fig

    def plot_spec(self, save=True, save_loc=f"out/spec_plots/"):
        """

        :param save:
        :param save_loc:
        :return:
        """
        # Error shading -- y +/- y_std/2
        e_wif = self.spec_wifes_err * 0.5

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12.15, 3))
        ax.plot(self.wave_wifes, self.spec_wifes_raw, 'b-',
                linewidth=0.75, label='WiFeS', zorder=2)
        ax.plot(self.wave_wifes, self.spec_wifes_err, 'r-',
                linewidth=0.55, label='Error', zorder=3)

        # Shade error
        ax.fill_between(self.wave_wifes, (self.spec_wifes_raw - e_wif), (self.spec_wifes_raw + e_wif),
                        alpha=0.3, facecolor='r')

        # Labels
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'Flux ($erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$)')
        ax.set_xlim([full_min, full_max])
        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout()

        # Save
        if save:
            plt.savefig(f'{save_loc}/{self.obj_name}.pdf')

        return fig


def dt2mjd(dt):
    """Matplotlib datetime to MJD"""
    dt = md.num2date(dt)
    x = Time(dt, format='datetime')
    x = x.to_value('mjd', 'float')
    return x


def red_blue_filename_sep(obj):
    """

    :param obj:
    :return:
    """
    result = {}
    for fn in obj['file']:
        fits_name = os.path.basename(fn)
        if fits_name.startswith("T2m3w"):
            if fits_name[5] == 'b':
                result['blue'] = fn
            elif fits_name[5] == 'r':
                result['red'] = fn
            else:
                print("Could not read file: " + fits_name)
        elif fits_name.startswith("OBK"):
            if 'blue' in fits_name.lower():
                result['blue'] = fn
            elif 'red' in fits_name.lower():
                result['red'] = fn
            else:
                print("Could not read file: " + fits_name)
        else:
            continue
    return pd.Series(result,
                     index=["blue", "red"],
                     dtype="object")


def make_amalgamated_file(raw_dir):
    """
    Read directory containing FITS files, or subdirectories containing FITS files
    and save a .csv file listing object and red/blue p11 FITS file paths

    :param raw_dir: path to directory containing FITS files
    :return: master list containing object and red/blue p11 FITS file paths
    """
    # Read .p11.fits files
    raw_file_list = glob.glob(f"{raw_dir}/**/*.p11.fits", recursive=True)
    print(f"Found {len(raw_file_list)} files.")

    # Generate object - file relation
    print(f"Generating {raw_dir}/object_fits_list.csv...")
    with open(f'{raw_dir}/object_fits_list.csv', 'w') as obj_list:
        obj_list.write('file,object\n')  # Column names
        for f in raw_file_list:
            fits_name = os.path.basename(f)
            if fits_name.startswith("T2m3w") or fits_name.startswith("OBK"):
                with fits.open(f) as hdu_l:
                    hdr = hdu_l[0].header
                    obj = hdr['OBJECT']
                    obj_list.write(f"{f},{obj}\n")
    print("Done")

    # Group red/blue file - object relation
    print(f"Condensing {raw_dir}/object_fits_list.csv...")
    obj_list = pd.read_csv(f'{raw_dir}/object_fits_list.csv')
    obj_list = obj_list.groupby(['object']).apply(red_blue_filename_sep).reset_index()
    obj_list.to_csv(f'{raw_dir}/object_fits_list.csv', index=False)
    print("Done")

    # Warning: Does not work well with duplicate object names in FITS headers
    print(f"Found {len(obj_list)} unique spectra.")

    return obj_list

# Testing
# spec_extract = SpecExtract("g0209537-135321",
#                            "../Data/CLAGNPlotter/raw_wifes/202211/T2m3wr-20221122.123045-0128.p11.fits",
#                            "../Data/CLAGNPlotter/raw_wifes/202211/T2m3wb-20221122.123045-0128.p11.fits")

# spec_extract.sky_aperture = 'annular'
# spec_extract.plot_spatial(save=False).show()
# spec_extract.generate_spec(save=False)
# spec_extract.plot_spec(save=False).show()
