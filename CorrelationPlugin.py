import numpy as np
from phy import IPlugin, connect
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvas
from phy.plot.visuals import PlotVisual, ScatterVisual
from phy.utils import emit, connect, unconnect, Bunch
import matplotlib.pyplot as plt

class CorrelationView(ManualClusteringView):
    plot_canvas_class = PlotCanvas

    def __init__(self, controller):
        super(CorrelationView, self).__init__()
        self.controller = controller
        self.colormap = plt.colormaps['plasma'].colors
        self.point_size = 50
        self.data_bounds = None
        self.cluster_ids = None

        self.canvas.set_layout('stacked', n_plots=1)
        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)
    
    def on_select(self, cluster_ids):
        pass

    def get_correlation_matrix(self):
        print('Computing correlation matrix...')
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

        correlation_matrix = np.zeros((len(self.cluster_ids), len(self.cluster_ids)))
        bin_width = 10 # ms
        for k in range(len(self.cluster_ids)):
            for j in range(k, len(self.cluster_ids)):
                if j == k:
                    correlation_matrix[k,j]=1
                    continue
                
                st1 = np.int32(spike_times[k]*1000/bin_width)
                s1 = np.zeros(np.max(st1)+1)
                s1[st1] = 1

                st2 = np.int32(spike_times[j]*1000/bin_width)
                s2 = np.zeros(np.max(st2)+1)
                s2[st2] = 1

                idx_end = np.min([len(s1), len(s2)])
                idx_start = np.max([st1[0], st2[0]])
                
                if idx_end<=idx_start:
                    correlation_matrix[k,j] = 0
                    correlation_matrix[j,k] = 0
                    continue
                
                s1 = s1[idx_start:idx_end]
                s2 = s2[idx_start:idx_end]

                correlation_matrix[k,j] = np.corrcoef(s1,s2)[0,1]
                correlation_matrix[j,k] = correlation_matrix[k,j]
            
            print(k, 'out of', len(self.cluster_ids), 'clusters done.')
        

        return correlation_matrix
    
    def get_color(self, corr):
        if np.isnan(corr):
            idx = 0
        else:
            n_colors = len(self.colormap)
            idx = np.int(np.floor(corr*(n_colors-1)))
            if idx<0:
                idx = 0
            elif idx >= n_colors:
                idx = n_colors-1
        
        color = np.concatenate((self.colormap[idx],np.array([1])))
        return color

    def refresh_matrix(self):
        self.correlation_matrix = self.get_correlation_matrix()
        self.point_size = 500/len(self.cluster_ids)
        self.refresh_figure()
    
    def refresh_figure(self):
        self.visual.reset_batch()
        x_plot = []
        y_plot = []
        c_plot = []
        for k in range(np.size(self.correlation_matrix,0)):
            for j in range(np.size(self.correlation_matrix,1)):
                x_plot.append(k)
                y_plot.append(j)
                c_plot.append(self.get_color(self.correlation_matrix[k,j]))

        x_plot = np.array(x_plot)
        y_plot = np.array(y_plot)
        c_plot = np.array(c_plot)

        self.data_bounds = (0, 0, np.size(self.correlation_matrix,0), np.size(self.correlation_matrix,1))

        self.visual.add_batch_data(
            x=x_plot, y=y_plot, color=c_plot, size=self.point_size, data_bounds=self.data_bounds, box_index=0)

        self.canvas.update_visual(self.visual)
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


class CorrelationViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_correlation_view():
            """A function that creates and returns a view."""
            view = CorrelationView(controller)

            @connect(sender=view)
            def on_view_attached(view_, gui):
                # NOTE: this callback function is called in CorrelationView.attach().
                @view.actions.add(prompt=False)
                def refresh_matrix():
                    """Refresh the view."""
                    view.refresh_matrix()
                @view.actions.add(prompt=True, prompt_default=lambda: str(view.point_size))
                def change_point_size(point_size):
                    """Change the point size displayed in the CorrelationView."""
                    view.point_size = point_size
                    view.refresh_figure()

                view.actions.separator()

            return view

        controller.view_creator['CorrelationView'] = create_correlation_view

