# Phy Waveform Plugin: remove strange waveforms in [phy](https://github.com/cortex-lab/phy)

## How to use it  
1. Create a myplugins.py file in `C:/Users/UserName/.phy/plugins/` and copy-paste the code from a plugin example on the GitHub repository  
2. Edit `C:/Users/UserName/.phy/phy_config.py`, and specify the plugin names to load in the GUI:  
```python
c = get_config()
c.TemplateGUI.plugins = ['SingleWaveformViewPlugin', 'SettingsPlugin']  # list of plugin names to load in the TemplateGUI
c.Plugins.dirs = [r'C:\Users\jiumao\.phy\plugins']
```

### Single Waveform View
![](doc/phy.png)
* Click Menubar -> View -> Add SingleWaveformView
* `Ctrl + Left Click` the waveform in `WaveformView` to select the channel
* `Ctrl + Click` to select the waveforms that intersect with the line you draw  
*Note*: `split` action will be faster with shorter line in width.  
![](doc/SingleWaveformView.png)
* Press `k` to split clusters or press `s` to split clusters and label the minor cluster as `noise`.  


## New shortcuts
* **s**: split clusters in `SingleWaveformView` and label the minor output cluster as `noise`. `Undo` action can't undo this action.
* **-**: defined in the `SettingsPlugin.py`. Apply the filter `ContamPct > 150`.

## About the plugins
* `WaveformPlugin.py`: defines the `SingleWaveformView`.
* `SettingsPlugin.py`: changes the columns in `ClusterView` and adds shortcut `-`.

## References
* [Customization in Phy](https://github.com/cortex-lab/phy/blob/master/docs/customization.md)
* [Phy Plugins](https://github.com/cortex-lab/phy/blob/master/docs/plugins.md)
