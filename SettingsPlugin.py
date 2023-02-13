from phy import IPlugin, connect

class SettingsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_controller_ready(sender):
            controller.supervisor.columns = ['id', 'ch', 'n_spikes', 'ContamPct', 'KSLabel']

        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            """This is called when the GUI and all objects are fully loaded.
            This is to make sure that controller.supervisor is properly defined.
            """

            @gui.view_actions.add(alias='contam')  # corresponds to `:fr` snippet
            def filter_ContamPct_alias(pct):
                """Filter clusters with the ContamPct."""
                controller.supervisor.filter('contam < %.1f' % float(pct))

            @controller.supervisor.actions.add(shortcut='-')
            def filter_ContamPct():
                """Filter clusters with the ContamPct."""
                controller.supervisor.filter('ContamPct<150')

            @connect(sender=controller.supervisor)
            def on_cluster(sender, up):
                """This is called every time a cluster assignment or cluster group/label
                changes."""
                print("Clusters update: %s" % up)

