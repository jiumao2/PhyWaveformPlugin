import numpy as np
from phy import IPlugin, connect
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvas
from phy.plot.visuals import PlotVisual, ScatterVisual, TextVisual
from phy.utils import emit, connect, unconnect, Bunch
import matplotlib.pyplot as plt
import scipy.signal as signal

class SimilarityMatrixView(ManualClusteringView):
    plot_canvas_class = PlotCanvas

    def __init__(self, controller):
        super(SimilarityMatrixView, self).__init__()
        self.controller = controller
        self.colormap = plt.colormaps['plasma'].colors
        self.point_size = 50
        self.data_bounds = None
        self.cluster_ids = None
        self.types = ['waveform', 'autocorrelogram', 'ISI']
        self.typeIndex = 0
        self.similarity_matrix_all = [None for _ in range(len(self.types))]
        self.dim = 1
        self.typeIndexDim2 = [0, 1]

        self.canvas.set_layout('stacked', n_plots=1)
        self.visual = ScatterVisual()
        self.text = TextVisual()
        self.canvas.add_visual(self.visual)
        self.canvas.add_visual(self.text)
    
    def on_select(self, cluster_ids):
        pass

    def get_similarity_matrix(self):
        print('Computing similarity matrix...')
        cluster_ids_all = self.controller.supervisor.clustering.cluster_ids
        groups = [self.controller.supervisor.get_labels('group')[cluster_id] for cluster_id in cluster_ids_all]
        idx = []
        for k in range(len(groups)):
            if groups[k] == 'good' or groups[k] == 'mua':
                idx.append(k)
        
        self.cluster_ids = cluster_ids_all[idx]
        channels = [self.controller.get_best_channel(self.cluster_ids[k]) for k in range(len(self.cluster_ids))]
        self.cluster_ids = self.cluster_ids[np.argsort(channels)]

        spike_times = [np.array(self.controller.get_spike_times(cluster_id)) for cluster_id in self.cluster_ids] # in seconds

        similarity_matrix = np.zeros((len(self.cluster_ids), len(self.cluster_ids)))
        if self.typeIndex == 0:
            n_waveform = 100
            waveform = []
            for k in range(len(self.cluster_ids)):        
                spike_ids = self.controller.selector(n_waveform, [self.cluster_ids[k]])
                data = self.controller.model.get_waveforms(spike_ids, None) # n_spikes, n_samples, n_channels

                if data is not None:
                    data = data - np.median(data, axis=1)[:, np.newaxis, :]

                assert data.ndim == 3  # n_spikes, n_samples, n_channels

                waveform_mean = np.mean(data, axis=0).transpose()
                waveform.append(waveform_mean.reshape(-1))
                print(k,'/',len(self.cluster_ids),'done')

            for k in range(len(self.cluster_ids)):
                for j in range(k+1, len(self.cluster_ids)):
                    cc = np.corrcoef(waveform[k], waveform[j])
                    similarity_matrix[k,j] = np.min([np.arctanh(cc[0,1]),10])
                    similarity_matrix[j,k] = similarity_matrix[k,j]   

        elif self.typeIndex == 1:
            cross_corr = []
            for k in range(len(self.cluster_ids)):
                st1 = np.int64(spike_times[k]*1000)
                s1 = np.zeros(np.max(st1)+1)
                s1[st1] = 1
                s2 = s1.copy()

                cc = np.zeros((101,))
                for j in range(50):
                    temp1 = s1[j:]
                    if j==0:
                        temp2 = s2
                    else:
                        temp2 = s2[:-j]
                    cc[j] = signal.correlate(temp1,temp2, mode='valid')
                    cc[100-j] = cc[j]
                
                cross_corr.append(cc)
                print(k,'/',len(self.cluster_ids),'done')
            
            for k in range(len(self.cluster_ids)):
                for j in range(k+1, len(self.cluster_ids)):
                    cc = np.corrcoef(cross_corr[k], cross_corr[j])
                    similarity_matrix[k,j] = np.min([np.arctanh(cc[0,1]),10])
                    similarity_matrix[j,k] = similarity_matrix[k,j]   

        elif self.typeIndex == 2:
            isi_freq = []
            for k in range(len(self.cluster_ids)):
                isi = np.diff(spike_times[k]*1000)
                isi_hist, _ = np.histogram(isi, np.arange(0, 100, 1))
                isi_freq.append(isi_hist/np.sum(isi_hist))
                print(k,'/',len(self.cluster_ids),'done')
            for k in range(len(self.cluster_ids)):
                for j in range(k+1, len(self.cluster_ids)):
                    cc = np.corrcoef(isi_freq[k], isi_freq[j])
                    similarity_matrix[k,j] = np.min([np.arctanh(cc[0,1]),10])
                    similarity_matrix[j,k] = similarity_matrix[k,j] 
        
        for k in range(len(self.cluster_ids)):
            similarity_matrix[k,k] = np.max(similarity_matrix, axis=(0,1))

        self.similarity_matrix = similarity_matrix
        self.similarity_matrix_all[self.typeIndex] = similarity_matrix
    
    def get_color(self, corr):
        corr = (corr-np.min(self.similarity_matrix))/(np.max(self.similarity_matrix)-np.min(self.similarity_matrix))
        if np.isnan(corr):
            idx = 0
        else:
            n_colors = len(self.colormap)
            idx = np.int64(np.floor(corr*(n_colors-1)))
            if idx<0:
                idx = 0
            elif idx >= n_colors:
                idx = n_colors-1
        
        color = np.concatenate((self.colormap[idx],np.array([1])))
        return color

    def refresh_matrix(self):
        self.get_similarity_matrix()
        self.point_size = 500/len(self.cluster_ids)
        self.refresh_figure()
    
    def refresh_figure(self):
        if self.dim == 1:
            self.refresh_figure_dim1()
        elif self.dim == 2:
            self.refresh_figure_dim2()

    def refresh_figure_dim1(self):
        self.similarity_matrix = self.similarity_matrix_all[self.typeIndex]
        if self.similarity_matrix is None:
            self.get_similarity_matrix()
            self.refresh_figure_dim1()
            return
        
        self.visual.reset_batch()
        self.text.reset_batch()
        
        x_plot = []
        y_plot = []
        c_plot = []
        for k in range(np.size(self.similarity_matrix,0)):
            for j in range(np.size(self.similarity_matrix,1)):
                x_plot.append(k)
                y_plot.append(j)
                c_plot.append(self.get_color(self.similarity_matrix[k,j]))

        x_plot = np.array(x_plot)
        y_plot = np.array(y_plot)
        c_plot = np.array(c_plot)

        self.data_bounds = (0, 0, np.size(self.similarity_matrix,0), np.size(self.similarity_matrix,1))

        self.visual.add_batch_data(
            x=x_plot, y=y_plot, color=c_plot, size=self.point_size, data_bounds=self.data_bounds, box_index=0)
        
        # Text
        self.text.add_batch_data(text=self.types[self.typeIndex], pos=np.array([0.5*len(self.cluster_ids),len(self.cluster_ids)]), box_index=0, anchor=(0,2), data_bounds=self.data_bounds)

        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.text)
        self.canvas.update()
    
    def refresh_figure_dim2(self):
        self.similarity_matrix1 = self.similarity_matrix_all[self.typeIndexDim2[0]]
        self.similarity_matrix2 = self.similarity_matrix_all[self.typeIndexDim2[1]]
        if self.similarity_matrix1 is None:
            self.typeIndex = self.typeIndexDim2[0]
            self.get_similarity_matrix()
            self.refresh_figure_dim2()
            return
        
        if self.similarity_matrix2 is None:
            self.typeIndex = self.typeIndexDim2[1]
            self.get_similarity_matrix()
            self.refresh_figure_dim2()
            return

        self.visual.reset_batch()
        self.text.reset_batch()
        
        cluster_id_x = []
        cluster_id_y = []
        x_plot = []
        y_plot = []
        for k in range(np.size(self.similarity_matrix1,0)):
            for j in range(np.size(self.similarity_matrix2,1)):
                if k==j:
                    continue
                cluster_id_x.append(k)
                cluster_id_y.append(j)
                x_plot.append(self.similarity_matrix1[k,j])
                y_plot.append(self.similarity_matrix2[k,j])

        self.x_plot = np.array(x_plot)
        self.y_plot = np.array(y_plot)
        self.cluster_id_x = np.array(cluster_id_x)
        self.cluster_id_y = np.array(cluster_id_y)

        self.data_bounds = (np.min(self.x_plot), np.min(self.y_plot), np.max(self.x_plot), np.max(self.y_plot))

        self.visual.add_batch_data(
            x=self.x_plot, y=self.y_plot, size=self.point_size, data_bounds=self.data_bounds, box_index=0)
        
        # Text
        self.text.add_batch_data(text=self.types[self.typeIndexDim2[0]], pos=np.array([0.5*(self.data_bounds[0]+self.data_bounds[2]),self.data_bounds[1]]), box_index=0, anchor=(0,-2), data_bounds=self.data_bounds)
        self.text.add_batch_data(text=self.types[self.typeIndexDim2[1]], pos=np.array([self.data_bounds[0],0.5*(self.data_bounds[1]+self.data_bounds[3])]), box_index=0, anchor=(-2,0), data_bounds=self.data_bounds)

        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.text)
        self.canvas.update()
    
    def on_mouse_click(self, e):
        if self.cluster_ids is None or self.data_bounds is None:
            return
        
        if 'Control' in e.modifiers:
            layout = getattr(self.canvas, 'layout', None)
            box_size_x, box_size_y = layout.box_size
            box, pos = layout.box_map(e.pos)

            x = pos[0] * box_size_x * (self.data_bounds[2] - self.data_bounds[0]) / 2 + (
                        self.data_bounds[0] + self.data_bounds[2]) / 2
            y = pos[1] * box_size_y * (self.data_bounds[3] - self.data_bounds[1]) / (1 + box_size_y) + (
                    self.data_bounds[3] - self.data_bounds[1]) / (1 + box_size_y) + self.data_bounds[1]

            print('Clicking', x, y)
            if self.dim==1:
                self.select_dim1(x, y)
            elif self.dim==2:
                self.select_dim2(x, y)
    
    def select_dim1(self, x, y):
        x = round(x)
        y = round(y)
        if x >= len(self.cluster_ids) or x < 0 or y >= len(self.cluster_ids) or y < 0:
            return
        
        print('id=='+str(self.cluster_ids[x])+' || id=='+str(self.cluster_ids[y]))
        if x==y:
            self.controller.supervisor.filter('id=='+str(self.cluster_ids[x]))
            emit('select', self.controller.supervisor, [self.cluster_ids[x]])
        else:
            self.controller.supervisor.filter('id=='+str(self.cluster_ids[x])+' || id=='+str(self.cluster_ids[y]))
            emit('select', self.controller.supervisor, [self.cluster_ids[x], self.cluster_ids[y]])
    
    def select_dim2(self, x, y):
        points = np.stack((self.x_plot, self.y_plot))
        idx = np.argmin(np.linalg.norm(points - np.array([x, y]).reshape((2,1)), axis=0))
        x = self.cluster_id_x[idx]
        y = self.cluster_id_y[idx]
        print('id=='+str(self.cluster_ids[x])+' || id=='+str(self.cluster_ids[y]))
        if x==y:
            self.controller.supervisor.filter('id=='+str(self.cluster_ids[x]))
            emit('select', self.controller.supervisor, [self.cluster_ids[x]])
        else:
            self.controller.supervisor.filter('id=='+str(self.cluster_ids[x])+' || id=='+str(self.cluster_ids[y]))
            emit('select', self.controller.supervisor, [self.cluster_ids[x], self.cluster_ids[y]])



class SimilarityViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_similarity_view():
            """A function that creates and returns a view."""
            view = SimilarityMatrixView(controller)

            @connect(sender=view)
            def on_view_attached(view_, gui):
                # NOTE: this callback function is called in SimilarityView.attach().
                @view.actions.add(prompt=False)
                def refresh_matrix():
                    """Refresh the view."""
                    view.refresh_matrix()
                @view.actions.add(prompt=True, prompt_default=lambda: str(view.point_size))
                def change_point_size(point_size):
                    """Change the point size displayed in the SimilarityView."""
                    view.point_size = point_size
                    view.refresh_figure()
                @view.actions.add(prompt=True, prompt_default=lambda: str(view.typeIndex))
                def change_type(typeIndex):
                    """Change the type displayed in the SimilarityView."""
                    view.typeIndex = typeIndex
                    view.refresh_figure()
                @view.actions.add(prompt=False)
                def next_type():
                    """Change to the next type displayed in the SimilarityView."""
                    view.typeIndex = (view.typeIndex+1)%len(view.types)
                    view.refresh_figure()
                
                view.actions.separator()
                @view.actions.add(prompt=False)
                def change_dim():
                    """Change the dimension displayed in the SimilarityView."""
                    view.dim = 3-view.dim
                    view.refresh_figure()
                @view.actions.add(prompt=True, prompt_default=lambda: str(view.typeIndexDim2))
                def change_type_dim2(typeIndex):
                    """Change the type displayed in the 2-dimension SimilarityView (Enter '0,1' or '0,2' or '1,2')."""
                    print('Type index:', typeIndex)
                    view.typeIndexDim2 = typeIndex
                    view.refresh_figure()

                view.actions.separator()

            return view

        controller.view_creator['SimilarityMatrixView'] = create_similarity_view

