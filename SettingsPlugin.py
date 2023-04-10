from phy import IPlugin, connect

class SettingsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_controller_ready(sender):
            controller.supervisor.columns = ['id', 'ch', 'n_spikes', 'ContamPct', 'KSLabel']

        # # Uncomment this section to add a new column 'myLabel' to the cluster view.
        # def get_group(cluster_id):
        #     """Return the cluster group."""
        #     try:
        #         group = controller.supervisor.get_labels('group')[cluster_id]
        #         if group == 'unsorted':
        #             group = None
        #         return group
        #     except:
        #         group_info = controller.model.metadata.get('group')
        #         if cluster_id in group_info.keys() and group_info['cluster_id'] != 'unsorted':
        #             return group_info['cluster_id']
        #         else:
        #             return None
        # 
        # controller.cluster_metrics['myLabel'] = get_group

        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            """This is called when the GUI and all objects are fully loaded.
            This is to make sure that controller.supervisor is properly defined.
            """

            @gui.view_actions.add(alias='contam')
            def filter_ContamPct_alias(pct):
                """Filter clusters with the ContamPct."""
                controller.supervisor.filter('contam < %.1f' % float(pct))

            @gui.view_actions.add(alias='good')
            def filter_good_alias():
                """Filter good clusters."""
                controller.supervisor.filter("group=='good'")

            @gui.view_actions.add(alias='mua')
            def filter_mua_alias():
                """Filter mua clusters."""
                controller.supervisor.filter("group=='mua'")

            @gui.view_actions.add(alias='noise')
            def filter_noise_alias():
                """Filter noise clusters."""
                controller.supervisor.filter("group=='noise'")

            @gui.view_actions.add(alias='muagood')
            def filter_muagood_alias():
                """Filter mua and good clusters."""
                controller.supervisor.filter("group=='mua' || group=='good'")

            @gui.view_actions.add(alias='goodmua')
            def filter_goodmua_alias():
                """Filter mua and good clusters."""
                controller.supervisor.filter("group=='mua' || group=='good'")

            @controller.supervisor.actions.add(shortcut='-')
            def filter_ContamPct():
                """Filter clusters with the ContamPct."""
                controller.supervisor.filter('ContamPct<150')

            @connect(sender=controller.supervisor)
            def on_cluster(sender, up):
                """This is called every time a cluster assignment or cluster group/label
                changes."""
                print("Clusters update: %s" % up)

