import numpy as np
from phy import IPlugin, connect
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvas
from phy.plot.visuals import PlotVisual
from phy.utils import emit, connect, unconnect


class SingleWaveformView(ManualClusteringView):
    plot_canvas_class = PlotCanvas

    def __init__(self, controller):
        super(SingleWaveformView, self).__init__()
        self.controller = controller
        self.waveform = None
        self.n_waveform = 300
        self.channel_id = None
        self.color = (0.03, 0.57, 0.98, .75)
        self.data_bounds = None

        self.canvas.set_layout('stacked', n_plots=1)
        self.visual = PlotVisual()
        self.canvas.add_visual(self.visual)

        self.line_x = np.zeros(2)
        self.line_y = np.zeros(2)
        self.count_points = 0

    def on_select(self, cluster_ids, **kwargs):
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return

        self.visual.reset_batch()
        self.cluster_ids = cluster_ids
        self.waveform = self.controller._get_waveforms_with_n_spikes(self.cluster_ids[0], self.n_waveform)

        self.channel_id = self.controller.get_best_channel(self.cluster_ids[0])

        y = self.waveform.data[:, :, self.waveform.channel_ids == self.channel_id]
        y = np.squeeze(y).transpose()
        x = np.arange(np.size(y, 0))

        self.data_bounds = (x[0], y.min(axis=(0, 1)), x[-1], y.max(axis=(0, 1)))

        for k in range(np.min((self.n_waveform, np.size(y, 1)))):
            self.visual.add_batch_data(
                x=x, y=y[:, k], color=self.color, data_bounds=self.data_bounds, box_index=0)

        self.canvas.update_visual(self.visual)
        self.canvas.update()

    def on_select_channel(self, waveformView=None, channel_id=None, key=None, button=None):
        if channel_id is None or not self.waveform:
            return

        self.visual.reset_batch()
        self.channel_id = channel_id
        y = self.waveform.data[:, :, self.waveform.channel_ids == self.channel_id]

        y = np.squeeze(y).transpose()
        x = np.arange(np.size(y, 0))

        self.data_bounds = (x[0], y.min(axis=(0, 1)), x[-1], y.max(axis=(0, 1)))

        for k in range(np.min((self.n_waveform, np.size(y, 1)))):
            self.visual.add_batch_data(
                x=x, y=y[:, k], color=self.color, data_bounds=self.data_bounds, box_index=0)

        self.canvas.update_visual(self.visual)
        self.canvas.update()

    def on_mouse_click(self, e):
        if not self.data_bounds:
            return
        if 'Control' in e.modifiers:
            layout = getattr(self.canvas, 'layout', None)
            box_size_x, box_size_y = layout.box_size
            box, pos = layout.box_map(e.pos)

            x = pos[0] * box_size_x * (self.data_bounds[2] - self.data_bounds[0]) / 2 + (
                        self.data_bounds[0] + self.data_bounds[2]) / 2
            y = pos[1] * box_size_y * (self.data_bounds[3] - self.data_bounds[1]) / (1 + box_size_y) + (
                    self.data_bounds[3] - self.data_bounds[1]) / (1 + box_size_y) + self.data_bounds[1]

            if self.count_points == 0:
                self.count_points = 1
                self.line_x[0] = x
                self.line_y[0] = y
            elif self.count_points == 1:
                self.count_points = 0
                self.line_x[1] = x
                self.line_y[1] = y
                self.draw_line()
            elif self.count_points >= 2:
                self.count_points = 1
                self.line_x[0] = x
                self.line_y[0] = y

    def draw_line(self):
        color = [1, 1, 1, 1]

        data = np.squeeze(self.waveform.data[:, :, self.waveform.channel_ids == self.channel_id])
        labels = np.zeros(np.size(data, 0))
        for k in range(np.size(data, 0)):
            for j in range(np.size(data, 1) - 1):
                if self.is_intersect(
                        np.array([self.line_x[0], self.line_y[0]]),
                        np.array([self.line_x[1], self.line_y[1]]),
                        np.array([j, data[k, j]]),
                        np.array([j + 1, data[k, j + 1]])
                ):
                    labels[k] = 1
                    continue

        self.visual.reset_batch()
        y = self.waveform.data[:, :, self.waveform.channel_ids == self.channel_id]
        y = np.squeeze(y).transpose()
        x = np.arange(np.size(y, 0))

        self.data_bounds = (x[0], y.min(axis=(0, 1)), x[-1], y.max(axis=(0, 1)))

        self.visual.add_batch_data(
            x=self.line_x, y=self.line_y, color=color, data_bounds=self.data_bounds, box_index=0)

        for k in range(np.min((self.n_waveform, np.size(y, 1)))):
            if labels[k] == 0:
                self.visual.add_batch_data(
                    x=x, y=y[:, k], color=self.color, data_bounds=self.data_bounds, box_index=0)
            else:
                self.visual.add_batch_data(
                    x=x, y=y[:, k], color=color, data_bounds=self.data_bounds, box_index=0)

        self.canvas.update_visual(self.visual)
        self.canvas.update()

    def is_intersect(self, P1, P2, Q1, Q2):
        if max(P1[0], P2[0]) < min(Q1[0], Q2[0]) \
                or max(Q1[0], Q2[0]) < min(P1[0], P2[0]) \
                or max(P1[1], P2[1]) < min(Q1[1], Q2[1]) \
                or max(Q1[1], Q2[1]) < min(P1[1], P2[1]):
            return False
        P1Q1 = np.zeros(3)
        P1P2 = np.zeros(3)
        P1Q2 = np.zeros(3)

        P1Q1[:2] = Q1 - P1
        P1P2[:2] = P2 - P1
        P1Q2[:2] = Q2 - P1
        P1Q1[2] = 0
        P1P2[2] = 0
        P1Q2[2] = 0
        a1 = np.cross(P1Q1, P1P2)
        a2 = np.cross(P1Q2, P1P2)
        if np.sign(a1[2] * a2[2]) >= 0:
            return False

        # swap P and Q, repeat the procedures
        temp = P1
        P1 = Q1
        Q1 = temp
        temp = P2
        P2 = Q2
        Q2 = temp

        P1Q1[:2] = Q1 - P1
        P1P2[:2] = P2 - P1
        P1Q2[:2] = Q2 - P1
        P1Q1[2] = 0
        P1P2[2] = 0
        P1Q2[2] = 0
        a1 = np.cross(P1Q1, P1P2)
        a2 = np.cross(P1Q2, P1P2)
        if np.sign(a1[2] * a2[2]) >= 0:
            return False

        return True

    def get_split_spike_ids(self):
        spike_ids = self.controller.get_spike_ids(self.cluster_ids[0])
        data = self.controller.model.get_waveforms(spike_ids, [self.channel_id])

        # Load the waveforms, either from the raw data directly, or from the _phy_spikes* files.
        if data is not None:
            data = data - np.median(data, axis=1)[:, np.newaxis, :]
        assert data.ndim == 3  # n_spikes, n_samples, n_channels

        # Filter the waveforms.
        if data is not None:
            data = self.controller.raw_data_filter.apply(data, axis=1)

        data = np.squeeze(data)
        ind = []

        x_start = min(self.line_x)
        x_end = max(self.line_x)
        for k in range(np.size(data, 0)):
            for j in range(np.max([0, np.int(np.floor(x_start))]),
                           np.min([np.size(data, 1) - 1, np.int(np.ceil(x_end))])
                           ):
                if self.is_intersect(
                        np.array([self.line_x[0], self.line_y[0]]),
                        np.array([self.line_x[1], self.line_y[1]]),
                        np.array([j, data[k, j]]),
                        np.array([j + 1, data[k, j + 1]])
                ):
                    ind.append(k)
                    continue

        return spike_ids[ind]

    def on_request_split(self, sender=None):
        return np.unique(self.get_split_spike_ids())

    def split_noise(self):
        spike_ids = self.get_split_spike_ids()
        if len(spike_ids) == 0:
            print('No spike selected!')
            return None

        return self.controller.supervisor.split(spike_ids)


class SingleWaveformViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_single_waveform_view():
            """A function that creates and returns a view."""
            view = SingleWaveformView(controller)

            connect(view.on_request_split)
            connect(view.on_select_channel)

            @connect
            def on_split_noise(sender):
                up = view.split_noise()
                if not up:
                    return
                idx0_count = np.sum(up.spike_clusters == up.added[0])
                idx1_count = np.sum(up.spike_clusters == up.added[1])

                old_cluster_id = up.deleted[0]
                noise_cluster_id = up.added[np.argmin([idx0_count, idx1_count])]
                good_cluster_id = up.added[np.argmax([idx0_count, idx1_count])]

                controller.supervisor.label('group', 'noise', cluster_ids=[noise_cluster_id])
                controller.supervisor.select(good_cluster_id)
                if good_cluster_id == old_cluster_id:
                    view.on_select(good_cluster_id)

            @connect(sender=view)
            def on_close_view(view, gui):
                unconnect(view.on_select_channel)
                unconnect(view.on_request_split)
                unconnect(on_split_noise)

            return view

        controller.view_creator['SingleWaveformView'] = create_single_waveform_view

        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            """This is called when the GUI and all objects are fully loaded.
            This is to make sure that controller.supervisor is properly defined.
            """

            @controller.supervisor.actions.add(shortcut='s')
            def split_noise():
                emit('split_noise', controller)
