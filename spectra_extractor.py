import subprocess
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time
from datetime import datetime
import matplotlib.dates as md
import glob
from matplotlib import rcParams

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

# set figure saving size/resolution
rcParams['savefig.bbox'] = 'tight'

#######################################

# Constants and Catalogues
n_row, n_col = 35, 25  # WiFeS cube face dimensions, remove 3 from n_row for bad pixel rows
r2 = (6.7 / 2) ** 2  # 6dFGS fibre radius squared
sep = 5650  # WiFeS red/blue separation wavelength
full_min, full_max = 4000, 7500  # WiFeS spectra range

# Emission line rest wavelengths
hbeta = 4861.35
halph = 6562.79
hgamm = 4340.47
O3 = 5007

# Lightcurve key dates
sensible_date_cutoff = 57900
bad_week = 59199
wall_1 = 58417
wall_2 = 58882

# Files and directories
cat_file = pd.read_csv("../6dFGS/Updated/6dFGS_all_final.csv")
agn_6df_file = pd.read_csv("../full_catalogue.csv")
dir_lightcurve = "lightcurves/"
dir_6dFGS = "6dFGS_raw/"
dir_all_plots = "plots/"


class SpecExtract:
    def __init__(self, gname, red_f, blue_f,
                 catalogue=cat_file, agn_6df=agn_6df_file,
                 dir_lc=dir_lightcurve, dir_6df=dir_6dFGS,
                 lc=False):
        self.gname = gname

        self.catalogue = catalogue
        self.agn_6df = agn_6df  # used only to get the type, don't need it for anything else.
        self.dir_lc = dir_lc
        self.dir_6df = dir_6df
        self.lc = lc

        print(f"Opening {gname}...")
        self.red_hdu = fits.open(red_f)
        self.blue_hdu = fits.open(blue_f)

        ''' Blue '''
        self.blue = self.blue_hdu[0].data[:, 3:, :]  # Remove bad pixel rows
        self.blue_e = self.blue_hdu[1].data[:, 3:, :]
        self.blueh = self.blue_hdu[0].header
        self.wave_blue = self.blueh['CRVAL3'] + np.arange(len(self.blue)) * self.blueh['CDELT3']
        self.blue_ind = np.ravel(np.argwhere((self.wave_blue > full_min) & (self.wave_blue <= sep)))
        self.blue_img = np.nanmedian(self.blue, axis=0)

        ''' Red '''
        self.red = self.red_hdu[0].data[:, 3:, :]  # Remove bad pixel rows
        self.red_e = self.red_hdu[1].data[:, 3:, :]
        self.redh = self.red_hdu[0].header
        self.wave_red = self.redh['CRVAL3'] + np.arange(len(self.red)) * self.redh['CDELT3']
        self.red_ind = np.ravel(np.argwhere((self.wave_red > sep) & (self.wave_red <= full_max)))
        self.red_img = np.nanmedian(self.red, axis=0)

        self.cat = self.catalogue[self.catalogue['targetname'] == self.gname].iloc[0]
        self.obsmjd = dt2mjd(md.date2num(datetime.fromisoformat(self.blueh['DATE-OBS'].split('T')[0])))
        self.ha = redshift_calc(self.cat['z'], halph)
        self.hb = redshift_calc(self.cat['z'], hbeta)
        self.hg = redshift_calc(self.cat['z'], hgamm)
        self.oo3 = redshift_calc(self.cat['z'], O3)

        self.row = 16
        self.col = 12
        self.row_min = 30
        self.col_min = 12
        self.mask = None
        self.mask_min = None

        self.spec_wifes_raw = None
        self.spec_wifes_err = None
        self.central_raw = None
        self.central_err = None
        self.wave_wifes = None

        self.red_hdu.close()
        self.blue_hdu.close()

    def make_masks(self, pos_list=None):
        if pos_list is not None:
            self.row = pos_list[0] + 3  # de-offset for bad rows at edge
            self.col = pos_list[1]
            self.row_min = pos_list[2] + 3  # de-offset for bad rows at edge
            self.col_min = pos_list[3]

        # Aperture mask
        x, y = np.ogrid[-self.row:n_row - self.row, -self.col:n_col - self.col]
        self.mask = x * x + y * y <= r2

        # Aperture mask for sky subtraction
        x, y = np.ogrid[-self.row_min:n_row - self.row_min, -self.col_min:n_col - self.col_min]
        self.mask_min = x * x + y * y <= r2

    def get_raw_files(self):
        if self.lc:
            if self.gname + '.lc' not in os.listdir(self.dir_lc):
                from_address_lc = "neelesh@mash.anu.edu.au:/mimsy/neelesh/honours/data/stack_7/"
                from_address_lc = from_address_lc + self.gname + '.lc'
                to_address_lc = self.dir_lc
                subprocess.run("scp " + from_address_lc + " " + to_address_lc, shell=True)

        if self.gname + '.fits' not in os.listdir(self.dir_6df):
            from_address_6df = "neelesh@mash.anu.edu.au:/priv/manta2/skymap/common/6dFGS/"
            from_address_6df = from_address_6df + f"{self.gname[1:3]}/{self.gname}.fits"
            to_address_6df = self.dir_6df
            subprocess.run("scp " + from_address_6df + " " + to_address_6df, shell=True)

    def save_spec(self, save_loc="WiFeS_with_error/", save=True):
        self.cat = self.catalogue[self.catalogue['targetname'] == self.gname].iloc[0]
        self.get_raw_files()
        del self.cat['fits1durl']
        self.cat['name'] = self.gname
        del self.cat['targetname']
        self.cat['row'] = self.row + 3  # de-offset while saving
        self.cat['row_sky'] = self.row_min + 3  # de-offset while saving
        self.cat['col'] = self.col
        self.cat['col_sky'] = self.col_min
        self.cat.dropna(inplace=True)

        # Average of sky in aperture
        blue_mean = np.mean(self.blue[:, self.mask_min], axis=1)
        red_mean = np.mean(self.red[:, self.mask_min], axis=1)
        blue_sky_sem = np.mean(self.blue_e[:, self.mask_min], axis=1) / np.sqrt(
            self.blue_e[:, self.mask_min].shape[1])
        red_sky_sem = np.mean(self.red_e[:, self.mask_min], axis=1) / np.sqrt(
            self.red_e[:, self.mask_min].shape[1])  # error

        # Get WiFeS spectra
        sqrt_n = np.sqrt(self.blue[:, self.mask].shape[1])
        spec_blue = np.sum(self.blue[:, self.mask] - blue_mean[:, None], axis=1)
        spec_red = np.sum(self.red[:, self.mask] - red_mean[:, None], axis=1)
        err_blue = np.sqrt(np.mean(self.blue_e[:, self.mask], axis=1)
                           / np.sqrt(self.blue_e[:, self.mask].shape[1])
                           + blue_sky_sem) * sqrt_n
        err_red = np.sqrt(np.mean(self.red_e[:, self.mask], axis=1)
                          / np.sqrt(self.red_e[:, self.mask].shape[1])
                          + red_sky_sem) * sqrt_n  # subtraction error

        self.spec_wifes_raw = np.concatenate((spec_blue[self.blue_ind], spec_red[self.red_ind]))
        self.spec_wifes_err = np.concatenate((err_blue[self.blue_ind], err_red[self.red_ind]))

        # Get WiFeS central pixel
        central_blue = self.blue[:, self.row, self.col] - blue_mean
        e_c_blue = np.sqrt(self.blue_e[:, self.row, self.col] + blue_sky_sem)  # error
        central_red = self.red[:, self.row, self.col] - red_mean
        e_c_red = np.sqrt(self.red_e[:, self.row, self.col] + red_sky_sem)  # error
        self.central_raw = np.concatenate((central_blue[self.blue_ind], central_red[self.red_ind]))
        self.central_err = np.concatenate((e_c_blue[self.blue_ind], e_c_red[self.red_ind]))

        self.wave_wifes = np.concatenate((self.wave_blue[self.blue_ind], self.wave_red[self.red_ind]))

        # MARZ-friendly FITS
        hdu = fits.PrimaryHDU(self.spec_wifes_raw)  # intensity array in primary
        hdu.header.update(self.cat)  # append object details dict to header
        hdu.header.update({"Data": "Intensity"})  # append object details dict to header
        hdu.header.update({"OBS_MJD": self.obsmjd})  # add observed MJD
        hdu1 = fits.ImageHDU(self.wave_wifes, name='wavelength')  # wavelength array in image extension
        hdu2 = fits.ImageHDU(self.central_raw, name='central')  # central pixel
        hdu3 = fits.ImageHDU(self.spec_wifes_err, name='int_err')  # intensity error
        hdu4 = fits.ImageHDU(self.central_err, name='central_err')  # central pixel error
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4])  # HDU list
        if save:
            hdul.writeto(f'{save_loc}/{self.gname}.fits', overwrite=True)  # write

    def plot_spatial(self, save=False, save_loc='spat_plots/'):
        circle1 = plt.Circle((self.col, self.row), 6.7 / 2, color='r', fill=False)
        circle2 = plt.Circle((self.col, self.row), 6.7 / 2, color='r', fill=False)
        circle3 = plt.Circle((self.col_min, self.row_min), 6.7 / 2, color='r', linestyle='--', fill=False)
        circle4 = plt.Circle((self.col_min, self.row_min), 6.7 / 2, color='r', linestyle='--', fill=False)
        rect1 = plt.Rectangle((self.col - 0.5, self.row - 0.5), 1, 1, color='r', fill=False)
        rect2 = plt.Rectangle((self.col - 0.5, self.row - 0.5), 1, 1, color='r', fill=False)

        fig, [axb, axr] = plt.subplots(1, 2)
        divider = make_axes_locatable(axb)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        imb = axb.imshow(self.blue_img)  # offset for bad rows at edge
        fig.colorbar(imb, cax=cax, orientation='vertical')
        axb.set_title('Blue')
        axb.add_patch(circle1)
        axb.add_patch(circle3)
        axb.add_patch(rect1)

        divider = make_axes_locatable(axr)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        imr = axr.imshow(self.red_img)  # offset for bad rows at edge
        fig.colorbar(imr, cax=cax, orientation='vertical')
        axr.set_title('Red')
        axr.add_patch(circle2)
        axr.add_patch(circle4)
        axr.add_patch(rect2)
        plt.tight_layout()

        if save:
            plt.savefig(save_loc + '/' + self.gname + '.pdf')

        return fig

    def plot_spec(self, text_add=True, save=True, save_loc=f"out/spec_plots/"):
        # Get 6dFGS spectra
        hdul = fits.open(f"{self.dir_6df}/{self.gname}.fits")
        spec_raw = hdul[1].data.reshape(-1)
        heads = hdul[1].header
        wave_6df = heads['CRVAL1'] + np.arange(heads['NAXIS1']) * heads['CDELT1']

        ind_o3_wifes = np.argmin(np.abs(self.wave_wifes - self.oo3)) - 10
        ind_o3_6df = np.argmin(np.abs(wave_6df - self.oo3)) - 10

        spec_raw = savgol_filter(spec_raw, 15, 3)
        spec_wifes_raw = savgol_filter(self.spec_wifes_raw, 15, 3)
        central_raw = savgol_filter(self.central_raw, 15, 3)

        o3_peak_wifes = np.argmax(spec_wifes_raw[ind_o3_wifes:ind_o3_wifes + 20]) + ind_o3_wifes
        o3_peak_central = np.argmax(central_raw[ind_o3_wifes:ind_o3_wifes + 20]) + ind_o3_wifes
        o3_peak_6df = np.argmax(spec_raw[ind_o3_6df:ind_o3_6df + 20]) + ind_o3_6df
        spec_wifes = spec_wifes_raw / spec_wifes_raw[o3_peak_wifes]
        spec_6df = spec_raw / spec_raw[o3_peak_6df]
        spec_central = central_raw / central_raw[o3_peak_central]

        y_text = 0.76
        agn_row = self.agn_6df[self.agn_6df["name"] == self.gname]

        if self.lc:
            lc = pd.read_csv("lightcurves/" + self.gname + '.lc', delim_whitespace=True)
            data = lc[(lc['###MJD'] > sensible_date_cutoff) & (lc['###MJD'] != bad_week)].reset_index(drop=True)
            c = data[data['F'] == 'c'].reset_index(drop=True)
            o = data[data['F'] == 'o'].reset_index(drop=True)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 2]})

            markers, caps, bars = ax1.errorbar(c["###MJD"], c["median"], yerr=(c["duJyl"], c["duJyu"]),
                                               color='darkturquoise', fmt='.', ms=5, label='Cyan')
            [bar.set_alpha(0.3) for bar in bars]
            markers, caps, bars = ax1.errorbar(o["###MJD"], o["median"], yerr=(o["duJyl"], o["duJyu"]),
                                               color='orange', fmt='.', ms=5, label='Orange')
            [bar.set_alpha(0.3) for bar in bars]

            ax1.axvline(x=wall_1, color='k', alpha=0.5)
            ax1.axvline(x=wall_2, color='k', alpha=0.5)
            ax1.axvline(x=self.obsmjd, color='k', linestyle='dashed')
            ax1.set_xlabel("Modified Julian Date")
            ax1.set_ylabel(r"Difference Flux ($\mu$Jy)")
            ax1.set_xlim([sensible_date_cutoff, 60060])
            ax1.legend(loc='upper left', fontsize=10)

            ax1.tick_params(axis='x', which='both', top=False)
            sec_ax = ax1.secondary_xaxis('top', functions=(mjd2dt, dt2mjd))
            sec_ax.set_xlabel('Year')
            sec_ax.xaxis.set_major_formatter(md.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
            sec_ax.tick_params(axis='x', which='minor', top=False)
            if text_add:
                text_to_ad3 = f"; SkyMapper Object ID: {self.cat['SMSS_id']:}; "
                text_to_add = "Object: " \
                              + self.cat + text_to_ad3 \
                              + f"\nRA: {self.cat['obsra']}; DEC: {self.cat['obsdec']}; "
                text_to_ad2 = f"z = {self.cat['z']:.5f};  6dFGS Type = {agn_row['type'].iat[0]}"
                ax1.text(0, 1.3, text_to_add + text_to_ad2, transform=ax1.transAxes)

        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(12.15, 3))

        ax2.plot(wave_6df, spec_6df, 'k-', linewidth=0.25, label='6dFGS', zorder=10)
        ax2.plot(self.wave_wifes, spec_wifes, 'b-', linewidth=0.75, label='WiFeS', zorder=2)
        ax2.plot(self.wave_wifes, spec_central, 'r-', linewidth=1, label='WiFeS Central', zorder=1)

        ax2.axvline(self.ha, alpha=0.4, color='#334f8d', linestyle='dotted')
        ax2.axvline(self.hb, alpha=0.4, color='#334f8d', linestyle='dotted')
        ax2.axvline(self.hg, alpha=0.4, color='#334f8d', linestyle='dotted')
        ax2.axvline(self.oo3, alpha=0.4, color='#114f8d', linestyle='dotted')

        ax2.text(x=self.ha + 5, y=y_text, s=r'H$\alpha$', alpha=0.7, color='#334f8d')
        ax2.text(x=self.hb + 5, y=y_text, s=r'H$\beta$', alpha=0.7, color='#334f8d')
        ax2.text(x=self.hg + 5, y=y_text, s=r'H$\gamma$', alpha=0.7, color='#334f8d')
        ax2.text(x=self.oo3 + 5, y=y_text, s=r'OIII', alpha=0.7, color='#114f8d')
        ax2.set_xlabel(r'Wavelength ($\AA$)')
        ax2.set_ylabel('Flux (Normalised)')
        ax2.set_xlim([4000, 7500])
        ax2.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        if save:
            plt.savefig(f'{save_loc}/{self.gname}.pdf')
        return fig


# Functions to convert MJD to matplotlib dates and back for plotting

def mjd2dt(mjd):
    """MJD to matplotlib datetime"""
    x = Time(mjd, format='mjd')
    x = x.to_value('datetime64', 'date')
    ys = pd.to_datetime(x)
    return [datetime(y.year, y.month, y.day) for y in ys]


def dt2mjd(dt):
    """Matplotlib datetime to MJD"""
    dt = md.num2date(dt)
    x = Time(dt, format='datetime')
    x = x.to_value('mjd', 'float')
    return x


def redshift_calc(z, wavelength):
    return wavelength * z + wavelength


def red_blue_filename_sep(obj):
    result = {}
    for fn in obj['file']:
        fits_name = os.path.basename(fn)
        if 'blue' in fits_name.lower():
            result['blue'] = fn
        else:
            result['red'] = fn
    return pd.Series(result,
                     index=["blue", "red"],
                     dtype="object")


def make_amalgamated_file(raw_dir):
    raw_file_list = glob.glob(f"{raw_dir}/*.fits")
    print(f"Found {len(raw_file_list)} files.")

    print(f"Generating {raw_dir}/object_fits_list.csv...")
    with open(f'{raw_dir}/object_fits_list.csv', 'w') as obj_list:
        obj_list.write('file,object\n')  # Column names
        for f in raw_file_list:
            with fits.open(f) as hdul:
                hdr = hdul[0].header
                obj = hdr['OBJECT']
                obj_list.write(f"{f},{obj}\n")
    print("Done")

    print(f"Condensing {raw_dir}/object_fits_list.csv...")
    obj_list = pd.read_csv(f'{raw_dir}/object_fits_list.csv')
    obj_list = obj_list.groupby(['object']).apply(red_blue_filename_sep).reset_index()
    obj_list.to_csv(f'{raw_dir}/object_fits_list.csv', index=False)
    print("Done")

    print(f"Found {len(obj_list)} unique spectra.")

    return obj_list

# spec_extract = SpecExtract("g0209537-135321",
#                            "raw_wifes/202211/T2m3wr-20221122.123045-0128.p11.fits",
#                            "raw_wifes/202211/T2m3wb-20221122.123045-0128.p11.fits")

# spec_extract.make_masks((13, 11, 30, 11))
# spec_extract.plot_spatial().show()
# spec_extract.save_spec(save=False)
# spec_extract.plot_spec(save=False).show()
