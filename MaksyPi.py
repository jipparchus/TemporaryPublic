import tkinter as tk
import tkinter.ttk as ttk

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import cv2
import pandas as pd
import PIL.Image, PIL.ImageTk
import subprocess
import threading


class Application(ttk.Notebook):
    def __init__(self, master=None):
        super().__init__(master)
        self.master.title('PiScope')
        self.master.geometry("1200x1000")
        
        tab1 = tk.Frame(self.master)
        self.add(tab1, text="Star Map")
        Tab1(master=tab1)

        tab2 = tk.Frame(self.master)
        self.add(tab2, text="Picture")
        Tab2(master=tab2)

        self._quit_app()
        self.pack()

    def _quit_app(self):
        quit = tk.Button(self.master, text="QUIT APP", command=root.destroy)
        quit.pack(side=tk.BOTTOM)

class Tab1(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_frames()
        self.date_location()
        self.plot_fig()
        self.pack()

    def create_frames(self):
        self.control_frame = tk.Frame(self)
        self.control_frame.pack(side=tk.TOP)
        self.canvas_frame = tk.Frame(self, relief=tk.SOLID, bd=5, width=20)
        self.canvas_frame.pack(side=tk.TOP)

        '''frames in control_frame'''
        self.date_frame = tk.Frame(self.control_frame, relief=tk.RIDGE, bd=5, width=20)
        self.date_frame.pack(side=tk.LEFT)
        self.location_frame = tk.Frame(self.control_frame, relief=tk.RIDGE, bd=5, width=20)
        self.location_frame.pack(side=tk.LEFT)

        '''frames in canvas_frame'''
        self.canvasctrl_frame = tk.Frame(self.canvas_frame, relief=tk.RIDGE, bd=5, width=20)
        self.canvasctrl_frame.pack(side=tk.RIGHT)
        self.canvasfig_frame = tk.Canvas(self.canvas_frame, width=1014, height=760)
        self.canvasfig_frame.pack(side=tk.RIGHT)

    def date_location(self):
        '''
        Date Entry (date_frame)
        '''
        now = datetime.now()
        label = tk.Label(self.date_frame, text='Local Date')
        label.grid(column=0, row=0)
        y_list = list(np.arange(1998, 2025, 1))
        self.y_cmb = ttk.Combobox(
            self.date_frame,
            values=y_list,
            state='readonly',
            width=4)
        self.y_cmb.set(now.year)
        self.y_cmb.grid(column=1, row=0)
        label = tk.Label(self.date_frame, text='/')
        label.grid(column=2, row=0)
        m_list = list(np.arange(1, 13, 1))
        self.m_cmb = ttk.Combobox(
            self.date_frame,
            values=m_list,
            state='readonly',
            width=2)
        self.m_cmb.set(now.month)
        self.m_cmb.grid(column=3, row=0)
        label = tk.Label(self.date_frame, text='/')
        label.grid(column=4, row=0)
        d_list = list(np.arange(1, 32, 1))
        self.d_cmb = ttk.Combobox(
            self.date_frame,
            values=d_list,
            state='readonly',
            width=2)
        self.d_cmb.set(now.day)
        self.d_cmb.grid(column=5, row=0)
        label = tk.Label(self.date_frame, text='Time')
        label.grid(column=6, row=0)
        h_list = list(np.arange(0, 24, 1))
        self.h_cmb = ttk.Combobox(
            self.date_frame,
            values=h_list,
            state='readonly',
            width=2)
        self.h_cmb.set(now.hour)
        self.h_cmb.grid(column=7, row=0)
        label = tk.Label(self.date_frame, text=':')
        label.grid(column=8, row=0)
        M_list = list(np.arange(0, 60, 1))
        self.M_cmb = ttk.Combobox(
            self.date_frame,
            values=M_list,
            state='readonly',
            width=2)
        self.M_cmb.set(now.minute)
        self.M_cmb.grid(column=9, row=0)
        label = tk.Label(self.date_frame, text=':')
        label.grid(column=10, row=0)
        S_list = list(np.arange(0, 60, 1))
        self.S_cmb = ttk.Combobox(
            self.date_frame,
            values=S_list,
            state='readonly',
            width=2)
        self.S_cmb.set(now.second)
        self.S_cmb.grid(column=11, row=0)

        label = tk.Label(self.date_frame, text='UTC offset [hour]')
        label.grid(column=0, row=1)
        pm = ['-','+']
        self.pm_cmb = ttk.Combobox(
            self.date_frame, 
            values=pm, 
            width=1, 
            state='readonly')
        self.pm_cmb.set('-')
        self.pm_cmb.grid(column=1, row=1, sticky=tk.E)
        utcoffset_list = list(np.arange(0,13,1))
        self.utcoffset_cmb = ttk.Combobox(
            self.date_frame, 
            values=utcoffset_list, 
            width=2, 
            state='readonly')
        self.utcoffset_cmb.current(0)
        self.utcoffset_cmb.grid(column=3, row=1, sticky=tk.W)
        label = tk.Label(self.date_frame, text='(* British summer time => -1 hour)')
        label.grid(column=12, row=1)
        
        '''
        Location Entry (location_frame)
        '''
        label = tk.Label(self.location_frame, text='Latitude [degree]')
        label.grid(column=0, row=0)
        self.lat_entry = tk.Entry(self.location_frame)
        self.lat_entry.insert(0, '52.404226036565326')
        self.lat_entry.grid(column=1, row=0)

        label = tk.Label(self.location_frame, text='Longitude [degree]')
        label.grid(column=0, row=1)
        self.lon_entry = tk.Entry(self.location_frame)
        self.lon_entry.insert(0, '-1.5210368')
        self.lon_entry.grid(column=1, row=1)
        label = tk.Label(self.location_frame, text='Height [m]')
        label.grid(column=0, row=2)
        self.hei_entry = tk.Entry(self.location_frame)
        self.hei_entry.insert(0, '89')
        self.hei_entry.grid(column=1, row=2)

        go_btn=tk.Button(self.location_frame, text='Go', command=lambda: self.get_datelocation)
        go_btn.grid(column=2, row=2)
        
    def get_datelocation(self):
        #CHECK IF DATE AND LOCATION ARE REASONABLE
        
        year = self.y_cmb.get()
        month = self.m_cmb.get()
        day = self.d_cmb.get()
        hour = self.h_cmb.get()
        minute = self.M_cmb.get()
        second = self.S_cmb.get()
        utcoffset_pm = self.pm_cmb.get()
        utcoffset = self.utcoffset_cmb.get()
        if len(month) == 1:
            month = '0' + str(month)
        if len(day) == 1:
            day = '0' + str(day)
        if len(hour) == 1:
            hour = '0' + str(hour)
        if len(minute) == 1:
            minute = '0' + str(minute)
        if len(second) == 1:
            second = '0' + str(second)

        localtime = year+'-'+month+'-'+day+' '+hour+':'+minute+':'+second        
        utcoffset = int(utcoffset_pm + str(utcoffset))*u.hour
        utcdate = Time(str(localtime)) + utcoffset

        lon = float(self.lon_entry.get())
        lat = float(self.lat_entry.get())
        height = float(self.hei_entry.get())
        earth_location = EarthLocation(lon=lon*u.deg, lat=lat*u.deg, height=height*u.m)
        
        print(utcdate, earth_location)
        return utcdate, earth_location        
        
    def plot_fig(self):
        '''
        Plot Figure (canvas_frame)
        '''
        fig = Figure()
        self.ax = fig.add_subplot(1, 1 , 1, polar=True)
        self.canvas = FigureCanvasTkAgg(fig, self.canvasfig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.messier_boolean = tk.BooleanVar()
        self.messier_boolean.set(True)
        self.messier_check = tk.Checkbutton(self.canvasctrl_frame, text='Messier Catelogue', variable=self.messier_boolean)
        self.messier_check.grid(column=0, row=0)

        show_btn=tk.Button(self.canvasctrl_frame, text='Show', command=self.draw_plot)
        show_btn.grid(column=0, row=1)
        
    def draw_plot(self):
        #initialise the axis
        self.ax.cla()
        self.ax.set_rlim([90, 0])
        self.ax.set_rgrids(np.arange(0, 91, 15), fontsize=9)
        self.ax.set_rlabel_position(-80)
        
        self.ax.set_theta_zero_location('N')
        self.ax.set_thetalim([2*np.pi, 0])
        self.ax.set_thetagrids(
            np.rad2deg(np.linspace(2*np.pi,0,9)[1:]),
            fontsize=12)
        self.ax.set_theta_direction(1)
        
        df_messier = self.messier_catalogue()

        if self.messier_boolean.get() == True:
            self.ax.scatter(np.deg2rad(df_messier['az']), df_messier['alt'], c='blue', s=3)
        
        else:
            #self.ax.cla()
            pass
        self.canvas.draw()

    def messier_catalogue(self):
        '''
        return Messier catalogue alt & az at given time and location as DF
        '''
        utcdate, location = self.get_datelocation()

        target_list = list(np.arange(1,111,1))
        target_list = ['M'+str(i) for i in target_list]

        df = pd.DataFrame()
        df['target'] = target_list

        alt_list = []
        az_list = []
        for t in target_list:
            target = SkyCoord.from_name(t)
            target_altaz = target.transform_to(AltAz(obstime=utcdate, location=location))
            az, alt = target_altaz.az.deg, target_altaz.alt.deg
            alt_list.append(alt)
            az_list.append(az)
        df['alt'] = alt_list
        df['az'] = az_list
        return df
        
class Tab2(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)       
        self.create_frames()
        self.capture_check()
        
        self.output = '/home/RPi4/MaksyPi/tmp/snapshot.jpg'
        
        
        self.capture()
        self.pack()
        
    def create_frames(self):
        self.frame1 = tk.Frame(self, relief=tk.SOLID, bd=1)
        self.frame1.pack(side=tk.TOP)
        self.frame2 = tk.Frame(self, relief=tk.SOLID, bd=1)
        self.frame2.pack(side=tk.TOP)
        self.frame3 = tk.Frame(self, relief=tk.SOLID, bd=1)
        self.frame3.pack(side=tk.TOP)
        
        '''frames in frame1'''
        #Zoom1
        self.frame1_1_canvas = tk.Canvas(self.frame1, relief=tk.SOLID, bd=3, width=200, height=200)
        self.frame1_1_canvas.pack(side=tk.LEFT)
        #Zoom2
        self.frame1_2_canvas = tk.Canvas(self.frame1, relief=tk.SOLID, bd=3, width=200, height=200)
        self.frame1_2_canvas.pack(side=tk.LEFT)
        #Zoom3
        self.frame1_3_canvas = tk.Canvas(self.frame1, relief=tk.SOLID, bd=3, width=200, height=200)
        self.frame1_3_canvas.pack(side=tk.LEFT)
        #Control
        self.frame1_4 = tk.Frame(self.frame1, relief=tk.RIDGE, bd=5)
        self.frame1_4.pack(side=tk.LEFT)

        '''frames in frame2'''
        #Main PreView
        self.frame2_1_canvas = tk.Canvas(self.frame2, relief=tk.SOLID, bd=5, width=761, height=570)
        self.frame2_1_canvas.pack(side=tk.TOP)
        #Control
        self.frame2_2 = tk.Frame(self.frame2, relief=tk.RIDGE, bd=5)
        self.frame2_2.pack(side=tk.TOP)
        
        '''frames in frame3'''
        #Residuals plot of Tracking
        self.frame3_1_canvas = tk.Canvas(self.frame3)
        self.frame3_1_canvas.pack(side=tk.TOP)
        #Details
        self.frame3_2 = tk.Frame(self.frame3, relief=tk.RIDGE, bd=5)
        self.frame3_2.pack(side=tk.BOTTOM)  
    
    def capture_check(self):
        label = tk.Label(self.frame2_2, text='Shutter Speed')
        label.grid(column=0, row=0)
        self.ss_entry = tk.Entry(self.frame2_2, width=5)
        self.ss_entry.insert(0, '100')
        self.ss_entry.grid(column=1, row=0)
        ss_list = ['ms', 'sec', 'min']
        self.ss_cmb = ttk.Combobox(
            self.frame2_2,
            values=ss_list,
            state='readonly',
            width=4)
        self.ss_cmb.set('ms')
        self.ss_cmb.grid(column=2, row=0)
        
        label = tk.Label(self.frame2_2, text='ISO')
        label.grid(column=3, row=0)
        iso_list = [100, 500, 1000, 1500, 2000, 3200, 6400, 12800]
        self.iso_cmb = ttk.Combobox(
            self.frame2_2,
            values=iso_list,
            state='readonly',
            width=4)
        self.iso_cmb.set(100)
        self.iso_cmb.grid(column=4, row=0)
        
        self.lview_boolean = tk.BooleanVar()
        self.lview_boolean.set(False)
        lview_check = tk.Checkbutton(self.frame2_2, text='Live View', variable=self.lview_boolean)
        lview_check.grid(column=0, row=1)
        
        btn = tk.Button(self.frame2_2, text='Free Memory', fg='red', command=lambda: self.free_memory())
        btn.grid(column=0, row=2)
        
    def set_ss(self):
        unit = self.ss_cmb.get()
        if unit == 'ms':
            ss_unit = 1000
        elif unit == 'sec':
            ss_unit = 1000000
        elif unit == 'min':
            ss_unit = 60000000
        self.shutter = int(self.ss_entry.get()) * ss_unit
        return self.shutter
    
    def capture(self):
        ss = self.set_ss()
        iso = self.iso_cmb.get()
        self.delay = ss/1000 + 20
    
        print(ss, iso)
        
        if self.lview_boolean.get() == True:            
            cmd = f"raspistill -ISO {iso} -ss {ss} -o {self.output} -n"
            subprocess.call(cmd.split())
                       
            self.img = PIL.Image.open('/home/RPi4/MaksyPi/tmp/snapshot.jpg')
            self.img = self.img.resize((761, 570))
            self.img = PIL.ImageTk.PhotoImage(self.img)
                
            self.frame2_1_canvas.create_image(5, 0, image=self.img, anchor=tk.NW)
            
        self.after_id = self.frame2_1_canvas.after(int(self.delay),  self.capture) 
        
    def free_memory(self):
        self.lview_boolean.set(False)
        self.frame2_1_canvas.delete('all')
        self.frame2_1_canvas.after_cancel(self.after_id)


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()


