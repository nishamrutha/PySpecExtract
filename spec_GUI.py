import matplotlib
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import pandas as pd
import spectra_extractor as se
import os

matplotlib.use('TkAgg')


class RawDirs:
    def __init__(self, root):
        self.raw_dir = None
        self.obj_list = None
        self.app = None
        self.catalogue = None
        self.agn_6df = None
        self.dir_lc = None
        self.dir_6df = None
        self.initial_dir = '/'

        self.root = root

        # Get raw dir
        self.raw_frame = Frame(self.root)
        self.root.title("Raw File Directory")
        self.raw_dir_log = Text(self.raw_frame, height=10)
        self.label_get_raw_dir = Label(self.raw_frame,
                                       text="Choose a directory with reduced WiFeS .fits files.")
        self.default_color = self.label_get_raw_dir['foreground']

        self.button_explore = Button(self.raw_frame,
                                     text="Select Directory",
                                     command=self.read_raw_dir)

        self.button_raw_dir_next = Button(self.root,
                                          text="Continue",
                                          command=self.raw_dir_next)

        self.label_get_raw_dir.pack()
        self.raw_dir_log.pack()
        self.button_explore.pack()
        sys.stdout.write = self.redirector  # whenever sys.stdout.write is called, redirector is called.

        # Catalogue (Jack's 6dFGS)
        self.cat_frame = Frame(self.root)
        self.label_cat = Label(self.cat_frame,
                               text="Catalogue:",
                               width=10)
        self.label_cat_txt = Label(self.cat_frame,
                                   text="Select 6dfgs_all_final.csv",
                                   width=50)
        self.button_cat = Button(self.cat_frame,
                                 text="Select",
                                 command=lambda: self.add_others('cat'))
        self.label_cat.pack(side=LEFT)
        self.label_cat_txt.pack(side=LEFT)
        self.button_cat.pack(side=LEFT)

        # agn_6df - full_catalogue.csv
        self.agn_frame = Frame(self.root)
        self.label_agn = Label(self.agn_frame,
                               text="AGN 6dF:",
                               width=10)
        self.label_agn_txt = Label(self.agn_frame,
                                   text="Select full_catalogue.csv",
                                   width=50)
        self.button_agn = Button(self.agn_frame,
                                 text="Select",
                                 command=lambda: self.add_others('agn'))
        self.label_agn.pack(side=LEFT)
        self.label_agn_txt.pack(side=LEFT)
        self.button_agn.pack(side=LEFT)

        # Lightcurves directory
        self.lc_frame = Frame(self.root)
        self.label_lc = Label(self.lc_frame,
                              text="Lightcurves:",
                              width=10)
        self.label_lc_txt = Label(self.lc_frame,
                                  text="Select lightcurve directory",
                                  width=50)
        self.button_lc = Button(self.lc_frame,
                                text="Select",
                                command=lambda: self.add_others('lc'))
        self.label_lc.pack(side=LEFT)
        self.label_lc_txt.pack(side=LEFT)
        self.button_lc.pack(side=LEFT)

        # 6dFGS spectra directory
        self.spc_frame = Frame(self.root)
        self.label_6df = Label(self.spc_frame,
                               text="6dF Spectra:",
                               width=10)
        self.label_6df_txt = Label(self.spc_frame,
                                   text="Select 6dF spectra directory",
                                   width=50)
        self.button_6df = Button(self.spc_frame,
                                 text="Select",
                                 command=lambda: self.add_others('6df'))
        self.label_6df.pack(side=LEFT)
        self.label_6df_txt.pack(side=LEFT)
        self.button_6df.pack(side=LEFT)

        self.raw_frame.pack()
        self.cat_frame.pack(pady=5, ipadx=5, padx=3)
        self.agn_frame.pack(pady=5, ipadx=5, padx=3)
        self.lc_frame.pack(pady=5, ipadx=5, padx=3)
        self.spc_frame.pack(pady=5, ipadx=5, padx=3)
        self.button_raw_dir_next.pack()

    def redirector(self, in_str):
        self.raw_dir_log.insert(INSERT, in_str)

    def raw_dir_next(self):
        """ Check for all fields and continue to the next window """
        next_ok = True
        if self.obj_list is None:
            self.label_get_raw_dir['foreground'] = 'red'
            next_ok = next_ok and False
        else:
            self.label_get_raw_dir['foreground'] = self.default_color

        if self.catalogue is None:
            self.label_cat_txt['foreground'] = 'red'
            next_ok = next_ok and False
        else:
            self.label_cat_txt['foreground'] = self.default_color

        if self.agn_6df is None:
            self.label_agn_txt['foreground'] = 'red'
            next_ok = next_ok and False
        else:
            self.label_agn_txt['foreground'] = self.default_color

        if self.dir_lc is None:
            self.label_lc_txt['foreground'] = 'red'
            next_ok = next_ok and False
        else:
            self.label_lc_txt['foreground'] = self.default_color

        if self.dir_6df is None:
            self.label_6df_txt['foreground'] = 'red'
            next_ok = next_ok and False
        else:
            self.label_6df_txt['foreground'] = self.default_color

        if next_ok:
            self.root.destroy()  # close the current window
            self.root = Tk()  # create another Tk instance
            self.app = MainWindow(self.root, self.raw_dir, self.obj_list, self.catalogue,
                                  self.agn_6df, self.dir_lc, self.dir_6df)
            self.root.mainloop()

    def read_raw_dir(self):
        """ Read the .fits files in the raw dir and make output dir structure """
        self.raw_dir = filedialog.askdirectory(initialdir=self.initial_dir,
                                               title="Select directory with reduced WiFeS cubes")

        self.obj_list = se.make_amalgamated_file(self.raw_dir)
        if len(self.obj_list) > 0:
            out_dir = f"{self.raw_dir}/out"
            for od in [out_dir, out_dir + "/WiFeS", out_dir + "/spec_plots", out_dir + "/spat_plots"]:
                if not os.path.exists(od):
                    os.makedirs(od)

        self.initial_dir = self.raw_dir
        self.label_get_raw_dir['foreground'] = self.default_color

    def add_others(self, other):
        """ Adding the different catalogues needed to complete the spectrum """
        if other == "cat":
            f = filedialog.askopenfilename(initialdir="../6dFGS/Updated/",
                                           title="Select 6dfgs_all_final.csv")
            self.catalogue = pd.read_csv(f)
            self.label_cat_txt['text'] = f
            self.label_cat_txt['foreground'] = self.default_color
            self.initial_dir = f
        elif other == "agn":
            f = filedialog.askopenfilename(initialdir="../",
                                           title="Select full_catalogue.csv")
            self.agn_6df = pd.read_csv(f)
            self.label_agn_txt['text'] = f
            self.label_agn_txt['foreground'] = self.default_color
            self.initial_dir = f
        elif other == "lc":
            self.dir_lc = filedialog.askdirectory(initialdir="lightcurves/",
                                                  title="Select lightcurve directory")
            self.label_lc_txt['text'] = self.dir_lc
            self.label_lc_txt['foreground'] = self.default_color
            self.initial_dir = self.dir_lc
        elif other == "6df":
            self.dir_6df = filedialog.askdirectory(initialdir="6dFGS_raw/",
                                                   title="Select 6dFGS spectra directory")
            self.label_6df_txt['text'] = self.dir_6df
            self.label_6df_txt['foreground'] = self.default_color
            self.initial_dir = self.dir_6df
        else:
            print("Select from [cat, agn, lc, 6df]")


class MainWindow:
    def __init__(self, root, raw_dir, obj_list, catalogue, agn_6df, dir_lc, dir_6df):
        sys.stdout.write = self.redirector  # whenever sys.stdout.write is called, redirector is called.
        self.raw_dir = raw_dir
        self.spat_plot_dir = raw_dir + "/out/spat_plots/"
        self.spec_plot_dir = raw_dir + "/out/spec_plots/"
        self.spec_data_dir = raw_dir + "/out/WiFeS/"
        self.obj_list = obj_list
        self.name_list = list(self.obj_list['object'])
        self.len_loaded = len(self.obj_list)
        self.catalogue = catalogue
        self.agn_6df = agn_6df
        self.dir_lc = dir_lc
        self.dir_6df = dir_6df
        self.root = root
        self.root.title("Spectra Extractor")
        self.counter = 0
        self.current_row = self.obj_list.iloc[self.counter]
        self.spec_object = None
        self.spat_plot_wid = None
        self.spat_canvas = None
        self.spat_toolbar = None
        self.spat_fig = None
        self.spec_fig = None
        self.spec_canvas = None
        self.spec_plot_wid = None
        self.spec_toolbar = None
        self.row = 16
        self.col = 12
        self.row_min = 30
        self.col_min = 12

        # Frames
        self.details_frame = Frame(self.root)
        self.spatial_frame = Frame(self.root, borderwidth=2, relief='sunken')
        self.spec_frame = Frame(self.root, borderwidth=2, relief='sunken')
        self.btn_frame = Frame(self.root, borderwidth=2, relief='sunken')

        # Log frame
        self.log_frame = Frame(self.root, borderwidth=2, relief='sunken')
        self.raw_dir_log = Text(self.log_frame, height=10)
        self.raw_dir_log.pack(fill=BOTH, expand=True)

        # Buttons
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
        self.opt_select = StringVar(self.btn_frame)
        self.opt_select.set(self.name_list[self.counter])
        self.opt_name = OptionMenu(self.btn_frame,
                                   self.opt_select, *self.name_list,
                                   command=self.opt_select_cmd)
        self.label_counter = Label(self.btn_frame,
                                   text=f"{self.counter + 1}/{self.len_loaded}")

        # Radio buttons to toggle sky and object position click event
        self.click_choice = StringVar(self.details_frame, "obj")
        self.label_click_select = Label(self.details_frame,
                                        text=f"Choose sky or object to move aperture on plot: ")
        self.radio_obj = Radiobutton(self.details_frame, text="Object", variable=self.click_choice, value='obj')
        self.radio_sky = Radiobutton(self.details_frame, text="Sky", variable=self.click_choice, value='sky')

        # Pack buttons in order
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

        # Pack the frames
        self.details_frame.pack(side=TOP)
        self.btn_frame.pack(side=BOTTOM)
        self.spec_frame.pack(fill=X, side=BOTTOM)
        self.log_frame.pack(fill=BOTH, side=RIGHT, expand=True)
        self.spatial_frame.pack()
        print(self.name_list)

        # Get the SpecExtract object for the current object and show initial plots
        self.spec_object = se.SpecExtract(self.current_row['object'], self.current_row['red'],
                                          self.current_row['blue'], agn_6df=self.agn_6df,
                                          dir_lc=self.dir_lc, dir_6df=self.dir_6df,
                                          lc=False)
        self.run_spec(save=False)

    def opt_select_cmd(self, choice):
        """ Option menu selector """
        self.counter = self.name_list.index(choice)
        self.label_counter['text'] = f"{self.counter + 1}/{self.len_loaded}"
        self.current_row = self.obj_list[self.obj_list['object'] == choice].iloc[0]
        self.reset_plots()
        self.spec_object = se.SpecExtract(self.current_row['object'], self.current_row['red'],
                                          self.current_row['blue'], agn_6df=self.agn_6df,
                                          dir_lc=self.dir_lc, dir_6df=self.dir_6df,
                                          lc=False)
        self.run_spec()

    def redirector(self, in_str):
        """ Print statements go to log text space """
        self.raw_dir_log.insert(INSERT, in_str)

    def clear_log(self):
        """ Clear the log """
        self.raw_dir_log.delete(1.0, END)

    def run_spec(self, save=False):
        """ Make spatial+spectral plot along with sky-subtracted data """
        self.spat_fig = self.spec_object.plot_spatial(save=save, save_loc=self.spat_plot_dir)
        self.spat_fig.canvas.callbacks.connect('button_press_event', self.get_row_col_click)
        self.spat_canvas = FigureCanvasTkAgg(self.spat_fig, master=self.spatial_frame)
        self.spat_fig.canvas.draw()
        self.spat_plot_wid = self.spat_canvas.get_tk_widget()
        self.spat_toolbar = NavigationToolbar2Tk(self.spat_canvas, self.spatial_frame)
        self.spat_toolbar.update()

        self.spec_object.make_masks()
        self.spec_object.save_spec(save_loc=self.spec_data_dir, save=save)

        self.spec_fig = self.spec_object.plot_spec(save=save, save_loc=self.spec_plot_dir)
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, master=self.spec_frame)
        self.spec_fig.canvas.draw()
        self.spec_plot_wid = self.spec_canvas.get_tk_widget()
        self.spec_toolbar = NavigationToolbar2Tk(self.spec_canvas, self.spec_frame)
        self.spec_toolbar.update()

        self.spat_plot_wid.pack(fill=BOTH, side=LEFT)
        self.spec_plot_wid.pack(fill=BOTH, side=LEFT)

    def save_current(self):
        """ Save the current spatial+spectral plots and spectral data """
        print(f"Saving {self.spec_object.gname}...")
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
            self.spec_object = se.SpecExtract(self.current_row['object'], self.current_row['red'],
                                              self.current_row['blue'], agn_6df=self.agn_6df,
                                              dir_lc=self.dir_lc, dir_6df=self.dir_6df,
                                              lc=False)
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
            self.spec_object = se.SpecExtract(self.current_row['object'], self.current_row['red'],
                                              self.current_row['blue'], agn_6df=self.agn_6df,
                                              dir_lc=self.dir_lc, dir_6df=self.dir_6df,
                                              lc=False)
            self.run_spec()

    def get_row_col_click(self, event):
        """ Get the object position """
        if event.inaxes is not None:
            if self.click_choice.get() == 'sky':
                self.col_min = int(event.xdata)
                self.row_min = int(event.ydata)
                print(f"Set new position for sky at {self.col_min}, {self.row_min}")
                self.update_spec_object()
                self.reset_plots()
                self.run_spec()
            else:
                self.col = int(event.xdata)
                self.row = int(event.ydata)
                print(f"Set new position for obj at {self.col}, {self.row}")
                self.col_min = int(event.xdata)
                self.row_min = 30
                print(f"Set new position for sky at {self.col_min}, {self.row_min}")
                self.update_spec_object()
                self.reset_plots()
                self.run_spec()
        else:
            print('Clicked outside axes bounds')

    def update_spec_object(self):
        """ Update the position of object and sky """
        self.spec_object.row = self.row
        self.spec_object.col = self.col
        self.spec_object.row_min = self.row_min
        self.spec_object.col_min = self.col_min

    def reset_plots(self):
        """ Destroy the previous plots """
        plt.close('all')
        self.spat_plot_wid.destroy()
        self.spat_toolbar.destroy()
        self.spec_plot_wid.destroy()
        self.spec_toolbar.destroy()


master = Tk()
# app = RawDirs(master)  # Run the app
app = MainWindow(master, "raw_wifes/thru20230416/",
                 pd.read_csv("raw_wifes/thru20230416/object_fits_list.csv"),
                 pd.read_csv("../6dFGS/Updated/6dfgs_all_final.csv"),
                 pd.read_csv("../full_catalogue.csv"),
                 "lightcurves/", "6dFGS_raw/")  # Testing
master.mainloop()
