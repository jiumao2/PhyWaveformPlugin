import numpy as np
import pandas as pd
from phy import IPlugin, connect
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvas
from phy.plot.visuals import PlotVisual, ScatterVisual, HistogramVisual, TextVisual
from phy.utils import emit, connect, unconnect, Bunch
import os
from neo.io import BlackrockIO

class PSTHView(ManualClusteringView):
    plot_canvas_class = PlotCanvas

    def __init__(self, controller):
        super(PSTHView, self).__init__()
        self.controller = controller
        self.color = (0.03, 0.57, 0.98, .75)
        self.color_axis = (.3, .3, .3, 1.)
        self.t_pre = -2000 # ms
        self.t_post = 2000 # ms
        self.binwidth = 50 # ms
        self.point_size = 2

        # self.event_labels = ['press', 'release', 'reward']
        self.event_idx = 0
        
        # Try to load the events file
        if not os.path.exists('events.csv') or not os.path.exists('event_labels.csv'):
            dir_nev = '..'
            if not os.path.exists(os.path.join(dir_nev, 'datafile001.nev')):
                dir_nev = os.path.join('..', '..')
            
            if not os.path.exists(os.path.join(dir_nev, 'datafile001.nev')):
                print('events.csv not found and cannot be created, please close PSTHview')
                return

            self.read_events(dir_nev)
        else:
            print('Found events.csv and event_labels.csv')

        events = pd.read_csv('events.csv',header=None)
        events_numpy = events.to_numpy()
        self.event_times = [events_numpy[i,~np.isnan(events_numpy[i,:])] for i in range(np.size(events_numpy,0))]

        self.event_labels = pd.read_csv('event_labels.csv',header=None).to_numpy().tolist()
        if len(self.event_labels[0])>1:
            self.event_labels = self.event_labels[0]
        else:
            self.event_labels = [self.event_labels[i][0] for i in range(len(self.event_labels))]

        self.canvas.set_layout('stacked', n_plots=2)
        self.raster = ScatterVisual()
        self.PSTH = HistogramVisual()
        self.axis = PlotVisual()
        self.text = TextVisual()

        self.canvas.add_visual(self.axis)
        self.canvas.add_visual(self.text)
        self.canvas.add_visual(self.raster)
        self.canvas.add_visual(self.PSTH)

    def read_events(self, dir_nev):
        block_index = []
        output = os.listdir(dir_nev)
        for file in output:
            if file.endswith('.nev'):
                print('Find nev file:', file[-7:-4])
                block_index.append(int(file[-7:-4]))

        block_index.sort()
        print('Block index:', block_index)
        print('Creating events.csv and event_labels.csv...')

        events = [[],[],[],[]]
        event_labels = ['press', 'trigger', 'release', 'reward']
        event_idx = [6,0,3,4]
        t0 = 0
        for idx_block in range(len(block_index)):
            print('Reading from', os.path.join(dir_nev, 'datafile00'+str(block_index[idx_block])+'.nev'),'...')
            bl = BlackrockIO(os.path.join(dir_nev, 'datafile00'+str(block_index[idx_block])+'.nev'))
            recording = BlackrockIO(os.path.join(dir_nev, 'datafile00'+str(block_index[idx_block])+'.ns6'))
            signal_size = recording.get_signal_size(block_index=0,seg_index=0,stream_index=1)

            t_event = bl.get_event_timestamps()
            timestamps = t_event[0]
            data_raw = t_event[2].astype(int)

            parsed_data = np.zeros((len(timestamps),7))
            for k in range(len(timestamps)):
                temp = '{:0>7}'.format(str(bin(data_raw[k]))[-7:])
                for j in range(7):
                    parsed_data[k,j] = int(temp[j])

            for idx in range(len(event_idx)):
                for k in range(1,len(timestamps)):
                    if parsed_data[k,event_idx[idx]]-parsed_data[k-1,event_idx[idx]] == 1:
                        events[idx].append(timestamps[k] + t0)
            
            t0 = t0+signal_size
                

        for idx in range(len(event_idx)):
            events[idx] = np.array(events[idx])/30000 # in seconds
            print(event_labels[idx], 'number', ':', len(events[idx]))

        pd.DataFrame(events).to_csv('events.csv',header=None,index=None)
        pd.DataFrame(event_labels).to_csv('event_labels.csv',header=None,index=None)
        print('Saved to events.csv and event_labels.csv')


    def on_select(self, cluster_ids, **kwargs):
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return

        self.raster.reset_batch()
        self.PSTH.reset_batch()
        self.axis.reset_batch()
        self.text.reset_batch()

        self.cluster_ids = cluster_ids
        spike_times = self.controller.get_spike_times(self.cluster_ids[0]) # sec

        event_times = self.event_times[self.event_idx]

        t_pre = self.t_pre/1000
        t_post = self.t_post/1000

        x_raster = []
        y_raster = []
        for k in range(np.size(event_times)):
            st = spike_times[spike_times>event_times[k]+t_pre]
            st = st[st<event_times[k]+t_post]
            for j in range(np.size(st)):
                x_raster.append(st[j]-event_times[k])
                y_raster.append(k)

        data_bounds_raster = (t_pre, 0, t_post, len(event_times))
        x_raster = np.array(x_raster)
        y_raster = np.array(y_raster)

        hist, bin_edges = np.histogram(x_raster, np.arange(t_pre, t_post, self.binwidth/1000))
        hist = hist*1000/self.binwidth/len(event_times)
        data_bounds_PSTH = (t_pre, 0, t_post, np.max(hist))

        # update the canvas

        # axis
        self.axis.add_batch_data(x=np.array([0,0]), y=np.array([0,len(event_times)]), box_index=0, color=self.color_axis, data_bounds=data_bounds_raster)
        self.axis.add_batch_data(x=np.array([t_pre,t_pre]), y=np.array([0,len(event_times)]), box_index=0, color=self.color_axis, data_bounds=data_bounds_raster)
        self.axis.add_batch_data(x=np.array([t_post,t_post]), y=np.array([0,len(event_times)]), box_index=0, color=self.color_axis, data_bounds=data_bounds_raster)

        self.axis.add_batch_data(x=np.array([t_pre,t_post]), y=np.array([0,0]), box_index=0, color=self.color_axis, data_bounds=data_bounds_raster)
        self.axis.add_batch_data(x=np.array([t_pre,t_post]), y=np.array([len(event_times),len(event_times)]), box_index=0, color=self.color_axis, data_bounds=data_bounds_raster)

        self.axis.add_batch_data(x=np.array([0,0]), y=np.array([0,np.max(hist)]), box_index=1, color=self.color_axis, data_bounds=data_bounds_PSTH)
        self.axis.add_batch_data(x=np.array([t_pre,t_pre]), y=np.array([0,np.max(hist)]), box_index=1, color=self.color_axis, data_bounds=data_bounds_PSTH)
        self.axis.add_batch_data(x=np.array([t_post,t_post]), y=np.array([0,np.max(hist)]), box_index=1, color=self.color_axis, data_bounds=data_bounds_PSTH)

        self.axis.add_batch_data(x=np.array([t_pre,t_post]), y=np.array([0,0]), box_index=1, color=self.color_axis, data_bounds=data_bounds_PSTH)
        self.axis.add_batch_data(x=np.array([t_pre,t_post]), y=np.array([np.max(hist),np.max(hist)]), box_index=1, color=self.color_axis, data_bounds=data_bounds_PSTH)

        # text
        text = [str(self.t_pre), '0', str(self.t_post)]
        text_pos = np.array([[t_pre,0],[0,0],[t_post,0]])
        self.text.add_batch_data(text=text, pos=text_pos, box_index=0, anchor=(0,-2), data_bounds=data_bounds_raster)
        self.text.add_batch_data(text='Time from '+self.event_labels[self.event_idx]+' (ms)', pos=np.array([0,0]), box_index=0, anchor=(0,-5), data_bounds=data_bounds_raster)

        self.text.add_batch_data(text='0', pos=np.array([t_pre,0]), box_index=0, anchor=(-2,0), data_bounds=data_bounds_raster)
        self.text.add_batch_data(text=str(len(event_times)), pos=np.array([t_pre,len(event_times)]), box_index=0, anchor=(-1,0), data_bounds=data_bounds_raster)

        self.text.add_batch_data(text='0', pos=np.array([t_pre,0]), box_index=1, anchor=(-2,0), data_bounds=data_bounds_PSTH)
        self.text.add_batch_data(text=str(round(np.max(hist))), pos=np.array([t_pre,np.max(hist)]), box_index=1, anchor=(-1,0), data_bounds=data_bounds_PSTH)

        # raster
        self.raster.add_batch_data(
            x=x_raster, y=y_raster, color=self.color, size=self.point_size, data_bounds=data_bounds_raster, box_index=0)

        # PSTH
        self.PSTH.add_batch_data(hist=hist, bin_edges=bin_edges, color=self.color, ylim=np.max(hist), data_bounds=data_bounds_PSTH, box_index=1)
        
        self.canvas.update_visual(self.axis)
        self.canvas.update_visual(self.text)
        self.canvas.update_visual(self.raster)
        self.canvas.update_visual(self.PSTH)
        self.canvas.update()


class PSTHViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_PSTH_view():
            """A function that creates and returns a view."""
            view = PSTHView(controller)

            @connect(sender=view)
            def on_view_attached(view_, gui):
                # NOTE: this callback function is called in PSTHView.attach().
                @view.actions.add(prompt=True, prompt_default=lambda: str(view.t_pre))
                def change_t_pre(t_pre):
                    """Change the t_pre in millisecond displayed in the PSTHView (enter positive number)."""
                    view.t_pre = -t_pre
                    view.on_select(view.cluster_ids) 

                @view.actions.add(prompt=True, prompt_default=lambda: str(view.t_post))
                def change_t_post(t_post):
                    """Change the t_post in millisecond displayed in the PSTHView."""
                    view.t_post = t_post
                    view.on_select(view.cluster_ids) 

                @view.actions.add(prompt=True, prompt_default=lambda: str(view.binwidth))
                def change_binwidth(binwidth):
                    """Change the binwidth in millisecond displayed in the PSTHView."""
                    view.binwidth = binwidth
                    view.on_select(view.cluster_ids) 

                view.actions.separator()

                @view.actions.add(prompt=True, prompt_default=lambda: str(view.event_idx))
                def change_event(event_idx):
                    """Change the event displayed in the PSTHView."""
                    view.event_idx = event_idx
                    view.on_select(view.cluster_ids) 

                @view.actions.add(prompt=False, alias='e')
                def next_event():
                    """Change to the next event displayed in the PSTHView."""
                    view.event_idx = np.mod(view.event_idx+1,len(view.event_labels))  
                    view.on_select(view.cluster_ids) 


            return view

        controller.view_creator['PSTHView'] = create_PSTH_view 

