#!/usr/bin/env python

"""
This module handles the GUI frontend of the PySpecExtract GUI designed
 to extract and analyze spectra from PyWiFeS-reduced ANU 2.3m telescope data.
 The code uses TkInter to graphically adjust parameters such as aperture size
 and position for sky subtraction to quickly browse and extract
 spectra from a large sample of .p11.fits files.
"""

#############
#  Imports  #
#############
import os
import sys
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

import spectra_extractor_psf as se

matplotlib.use('TkAgg')

##############
# Authorship #
##############
__author__ = "Neelesh Amrutha"
__date__ = "03 December 2025"

__license__ = "GPL-3.0"
__version__ = "2.0"
__maintainer__ = "Neelesh Amrutha"
__email__ = "neelesh.amrutha<AT>anu.edu.au"


class RawDirs:
    """
    Class representing a directory selection window.

    Parameters:
        root: TkInter root object for the GUI.

    Attributes:
        raw_dir: The selected raw file directory.
        obj_list: The list of object and blue/red filepaths.
        app: Instance of the MainWindow class for the main GUI window.
        initial_dir: Initial directory path.
        root: TkInter root object for the GUI.
        raw_frame: Frame widget for the directory selection window.
        raw_dir_log: Text widget for displaying log messages.
        label_get_raw_dir: Label widget displaying a prompt to choose a directory.
        default_color: Default color of the label_get_raw_dir text.

        button_explore: Button widget to open the file explorer.
        button_raw_dir_next: Button widget to continue to the next window.

    Methods:
        text_redirector: Redirects print statements to the GUI log.
        raw_dir_next: Checks for valid object list file and proceeds to the next window.
        read_raw_dir: Reads the .p11.fits files in the input directory
                    and creates output directory structure.
    """

    def __init__(self, root):
        self.raw_dir = None
        self.obj_list = None
        self.app = None
        self.initial_dir = '/'

        # TkInter App root
        self.root = root

        # Arrange directory selection window
        self.raw_frame = Frame(self.root)
        self.root.title("Raw File Directory")
        self.raw_dir_log = Text(self.raw_frame, height=10)
        self.label_get_raw_dir = Label(self.raw_frame,
                                       text="Choose a directory with reduced WiFeS .fits files.")
        self.default_color = self.label_get_raw_dir['foreground']

        # Button to open file explorer
        self.button_explore = Button(self.raw_frame,
                                     text="Select Directory",
                                     command=self.read_raw_dir)

        # Button to continue
        self.button_raw_dir_next = Button(self.root,
                                          text="Continue",
                                          command=self.raw_dir_next)

        self.label_get_raw_dir.pack()
        self.raw_dir_log.pack()
        self.button_explore.pack()
        sys.stdout.write = self.text_redirector  # the new print function
        self.raw_frame.pack()
        self.button_raw_dir_next.pack()

    def text_redirector(self, in_str):
        """ Channels print statements to GUI log """
        self.raw_dir_log.insert(INSERT, in_str)
        self.raw_dir_log.see("end")

    def raw_dir_next(self):
        """ Check for object list file and continue to the next window """
        if self.obj_list is None:
            self.label_get_raw_dir['foreground'] = 'red'
        else:
            # Clear the current window
            for w in self.root.winfo_children():
                w.destroy()

            # Start the main GUI window
            self.app = MainWindow(self.root, self.raw_dir, self.obj_list)

    def read_raw_dir(self):
        """ Read the .p11.fits files in inout directory and make output dir structure """
        self.raw_dir = filedialog.askdirectory(initialdir=self.initial_dir,
                                               title="Select directory with reduced WiFeS cubes")

        # Generate object_fits_list.csv
        self.obj_list = se.make_amalgamated_file(self.raw_dir)

        if len(self.obj_list) > 0:
            out_dir = f"{self.raw_dir}/out"

            # Output directory structure
            for od in [out_dir, out_dir + "/WiFeS", out_dir + "/spec_plots", out_dir + "/spat_plots"]:
                if not os.path.exists(od):
                    os.makedirs(od)
            print(f"Output directory: {out_dir}/")
        else:
            print("No .p11.fits files found. Choose another directory.")

        self.initial_dir = self.raw_dir
        self.label_get_raw_dir['foreground'] = self.default_color


class MainWindow:
    """
    Main window class for PySpecExtract GUI
    """

    def __init__(self, root, raw_dir, obj_list):
        """
        Initialize the MainWindow class.

        Parameters:
            root (Tk): Root window object.
            raw_dir (str): Top-level directory containing FITS files.
            obj_list (pandas.DataFrame): Objects and corresponding blue/red filepaths.
        """
        sys.stdout.write = self.text_redirector  # new print function

        # Directories
        self.raw_dir = raw_dir
        self.spat_plot_dir = raw_dir + "/out/spat_plots/"
        self.spec_plot_dir = raw_dir + "/out/spec_plots/"
        self.spec_data_dir = raw_dir + "/out/WiFeS/"
        self.wave_prof_dir = raw_dir + "/out/WiFeS/wave_profiles/"

        # Object list
        self.obj_list = obj_list
        self.name_list = list(self.obj_list['object'])
        self.len_loaded = len(self.obj_list)

        # App root
        self.root = root
        self.root.title("PySpecExtract")

        # Track iteration
        self.counter = 0
        self.current_row = self.obj_list.iloc[self.counter]

        # Plot widgets
        self.spat_plot_wid = None
        self.spat_canvas = None
        self.spat_toolbar = None
        self.spat_fig = None
        self.spec_fig = None
        self.spec_canvas = None
        self.spec_plot_wid = None
        self.spec_toolbar = None

        # SpecExtract object parameters
        self.spec_object = None
        self.row = 16
        self.col = 12
        self.row_min = 30
        self.col_min = 12
        self.r = 2  # aperture radius
        self.sky_aperture = 'annular'  # [disjoint, annular]
        self.sky_r = 2  # sky aperture radius

        # Frames
        self.obj_sky_frame = Frame(self.root)
        self.spatial_frame = Frame(self.root, borderwidth=2, relief='sunken')
        self.spec_frame = Frame(self.root, borderwidth=2, relief='sunken')
        self.btn_frame = Frame(self.root, borderwidth=2, relief='sunken')
        self.log_frame = Frame(self.root, borderwidth=2, relief='sunken')

        # Log
        self.raw_dir_log = Text(self.log_frame, height=10)
        self.raw_dir_log.pack(fill=BOTH, expand=True)

        # Buttons to save and navigate objects
        self.btn_exit = Button(self.btn_frame,
                               text="Exit", command=exit)
        self.btn_clr_log = Button(self.btn_frame,
                                  text="Clear log", command=self.clear_log)
        self.btn_save = Button(self.btn_frame,
                               text="Save", command=self.save_current)
        self.btn_next = Button(self.btn_frame,
                               text="Next >", command=self.next_cmd)
        self.btn_back = Button(self.btn_frame,
                               text="< Back", command=self.back_cmd)

        # Option menu to select specific object
        self.opt_select = StringVar(self.btn_frame)
        self.opt_name = OptionMenu(self.btn_frame, self.opt_select,
                                   self.name_list[self.counter], *self.name_list,
                                   command=self.opt_select_cmd)
        self.label_counter = Label(self.btn_frame,
                                   text=f"{self.counter + 1}/{self.len_loaded}")

        # Radio buttons to toggle sky and object position click event
        self.click_choice = StringVar(self.obj_sky_frame, "obj")
        self.label_click_select = Label(self.obj_sky_frame,
                                        text=f"Click on plot to change: ")
        self.radio_obj = Radiobutton(self.obj_sky_frame, text="Object", variable=self.click_choice, value='obj')
        self.radio_sky = Radiobutton(self.obj_sky_frame, text="Sky", variable=self.click_choice, value='sky')

        # Radio buttons to toggle sky aperture
        self.sky_choice = StringVar(self.obj_sky_frame, self.sky_aperture)
        self.label_sky_select = Label(self.obj_sky_frame,
                                      text="Sky type: ")
        self.radio_sky_free = Radiobutton(self.obj_sky_frame,
                                          text="Free", variable=self.sky_choice, value='disjoint',
                                          command=self.change_sky_aperture)
        self.radio_sky_ann = Radiobutton(self.obj_sky_frame,
                                         text="Annular", variable=self.sky_choice, value='annular',
                                         command=self.change_sky_aperture)

        # Entry box to set object and sky aperture size
        self.r_apt_var = DoubleVar(self.obj_sky_frame, self.r)
        self.r_sky_var = DoubleVar(self.obj_sky_frame, self.sky_r)
        self.label_r_apt = Label(self.obj_sky_frame, text="Aperture R:")
        self.entry_r_apt = Entry(self.obj_sky_frame, textvariable=self.r_apt_var, width=5)
        self.label_r_sky = Label(self.obj_sky_frame, text="Sky R/width:")
        self.entry_r_sky = Entry(self.obj_sky_frame, textvariable=self.r_sky_var, width=5)
        self.btn_r_entry = Button(self.obj_sky_frame, text="Enter", command=self.enter_r_apt_sky)

        # Pack buttons in order (using .pack instead of .grid was a bad idea)
        self.label_counter.pack(side=LEFT)
        self.btn_exit.pack(side=LEFT)
        self.btn_clr_log.pack(side=LEFT)
        self.btn_save.pack(side=LEFT)
        self.btn_back.pack(side=LEFT)
        self.btn_next.pack(side=LEFT)
        self.opt_name.pack(side=LEFT)
        self.label_click_select.pack(side=LEFT)
        self.radio_obj.pack(side=LEFT)
        self.radio_sky.pack(side=LEFT)
        self.label_sky_select.pack(side=LEFT, padx=(30, 1))
        self.radio_sky_ann.pack(side=LEFT)
        self.radio_sky_free.pack(side=LEFT)
        self.label_r_apt.pack(side=LEFT, padx=(30, 1))
        self.entry_r_apt.pack(side=LEFT)
        self.label_r_sky.pack(side=LEFT, padx=(10, 1))
        self.entry_r_sky.pack(side=LEFT)
        self.btn_r_entry.pack(side=LEFT)

        # Pack the frames
        self.obj_sky_frame.pack(side=TOP)
        self.btn_frame.pack(side=BOTTOM)
        self.spec_frame.pack(fill=X, side=BOTTOM)
        self.log_frame.pack(fill=BOTH, side=RIGHT, expand=True)
        self.spatial_frame.pack()

        print(f"Loaded {self.len_loaded} objects.")

        # Bind RETURN key to aperture size command
        self.root.bind('<Return>', self.enter_r_apt_sky)

        # Get the SpecExtract object for the current object and show plots
        self.spec_object = self.get_new_spec_object()
        self.run_spec(save=False)

    def opt_select_cmd(self, choice):
        """ Option menu selector """
        self.counter = self.name_list.index(choice)
        self.label_counter['text'] = f"{self.counter + 1}/{self.len_loaded}"
        self.current_row = self.obj_list[self.obj_list['object'] == choice].iloc[0]
        self.reset_plots()
        self.spec_object = self.get_new_spec_object()
        self.run_spec()

    def text_redirector(self, in_str):
        """ Redirect print statements to log text space """
        self.raw_dir_log.insert(INSERT, in_str)
        self.raw_dir_log.see("end")

    def clear_log(self):
        """ Clear the log """
        self.raw_dir_log.delete(1.0, END)

    def run_spec(self, save=False):
        """ Generate spatial+spectral plot along with sky-subtracted data """

        # Plot spatial image
        self.spat_fig = self.spec_object.plot_spatial(save=save, save_loc=self.spat_plot_dir)
        self.spat_fig.canvas.callbacks.connect('button_press_event', self.get_row_col_click)
        self.spat_canvas = FigureCanvasTkAgg(self.spat_fig, master=self.spatial_frame)
        self.spat_fig.canvas.draw()
        self.spat_plot_wid = self.spat_canvas.get_tk_widget()
        self.spat_toolbar = NavigationToolbar2Tk(self.spat_canvas, self.spatial_frame)
        self.spat_toolbar.update()

        # Generate spectrum
        self.spec_object.make_masks()
        self.spec_object.generate_spec(save_loc=self.spec_data_dir, save=save)

        # Plot spectrum
        self.spec_fig = self.spec_object.plot_spec(save=save, save_loc=self.spec_plot_dir)
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, master=self.spec_frame)
        self.spec_fig.canvas.draw()
        self.spec_plot_wid = self.spec_canvas.get_tk_widget()
        self.spec_toolbar = NavigationToolbar2Tk(self.spec_canvas, self.spec_frame)
        self.spec_toolbar.update()

        # Pack images
        self.spat_plot_wid.pack(fill=BOTH, side=LEFT)
        self.spec_plot_wid.pack(fill=BOTH, side=LEFT)

    def save_current(self):
        """ Save the current spatial+spectral plots and spectral data """
        print(f"Saving {self.spec_object.obj_name}...")
        self.reset_plots()
        self.run_spec(save=True)
        print("Saved in " + self.raw_dir + "/out/")

    def next_cmd(self):
        """ Go forward in the object list by one step """
        if self.counter == self.len_loaded - 1:
            print("End of directory. Try going back.")
        else:
            self.counter = self.counter + 1
            self.label_counter['text'] = f"{self.counter + 1}/{self.len_loaded}"
            self.current_row = self.obj_list[self.obj_list['object'] == self.name_list[self.counter]].iloc[0]
            self.reset_plots()
            self.spec_object = self.get_new_spec_object()
            self.run_spec()

    def back_cmd(self):
        """ Go back in the object list by one step """
        if self.counter == 0:
            print("Beginning of directory. Try going forward.")
        else:
            self.counter = self.counter - 1
            self.label_counter['text'] = f"{self.counter + 1}/{self.len_loaded}"
            self.current_row = self.obj_list[self.obj_list['object'] == self.name_list[self.counter]].iloc[0]
            self.reset_plots()
            self.spec_object = self.get_new_spec_object()
            self.run_spec()

    def get_row_col_click(self, event):
        """ Get object position from click events and update plots """
        if event.inaxes is not None:  # click inside axes
            if self.click_choice.get() == 'sky':
                if self.sky_aperture == 'disjoint':
                    self.col_min = round(event.xdata)
                    self.row_min = round(event.ydata)
                    print(f"Set new position for sky at {self.col_min}, {self.row_min}")
                    self.update_spec_object()
                else:
                    print("Select free sky aperture to choose sky region.")
            else:  # Object
                self.col = round(event.xdata)
                self.row = round(event.ydata)
                print(f"Set new position for obj at {self.col}, {self.row}")
                self.update_spec_object()
        else:
            print('Clicked outside axes bounds')

    def change_sky_aperture(self):
        """ Toggle free and annular sky aperture and update plots """
        self.sky_aperture = self.sky_choice.get()
        self.update_spec_object()

    def enter_r_apt_sky(self, _=None):
        """ Set object and sky aperture radius and update plots """
        self.r = self.r_apt_var.get()
        self.sky_r = self.r_sky_var.get()
        self.update_spec_object()

    def update_spec_object(self):
        """ Update SpecExtract and GUI plots """
        self.spec_object.row = self.row
        self.spec_object.col = self.col
        self.spec_object.row_min = self.row_min
        self.spec_object.col_min = self.col_min
        self.spec_object.r = self.r
        self.spec_object.sky_aperture = self.sky_aperture
        self.spec_object.sky_r = self.sky_r
        self.reset_plots()
        self.run_spec()

    def get_new_spec_object(self):
        """ Generate new SpecExtract object """
        return se.SpecExtract(self.current_row['object'],
                              self.current_row['red'],
                              self.current_row['blue'],
                              r=self.r,
                              sky_aperture=self.sky_aperture,
                              sky_r=self.sky_r,
                              row=self.row,
                              col=self.col,
                              row_min=self.row_min,
                              col_min=self.col_min)

    def reset_plots(self):
        """ Destroy the previous plots """
        plt.close('all')
        self.spat_plot_wid.destroy()
        self.spat_toolbar.destroy()
        self.spec_plot_wid.destroy()
        self.spec_toolbar.destroy()


# Start the application
master = Tk()
app = RawDirs(master)
# app = MainWindow(master, "../Data/CLAGNPlotter/raw_wifes/",
#                  pd.read_csv("../Data/CLAGNPlotter/raw_wifes/object_fits_list.csv"))  # Testing
master.mainloop()
