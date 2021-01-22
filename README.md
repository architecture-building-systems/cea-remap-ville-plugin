# CEA Plugin for the ReMaP project

To install, clone this repo to a desired path (you would need to have `git` installed to run this command. 
Alternatively you can also run this command in the CEA console):

```git clone https://github.com/architecture-building-systems/cea-remap-ville-plugin.git DESIRED_PATH```


Open CEA console and enter the following command to install the plugin to CEA:

```pip install -e PATH_OF_PLUGIN_FOLDER```

(NOTE: PATH_OF_PLUGIN_FOLDER would be the DESIRED_PATH + 'cea-remap-ville-plugin')


In the CEA console, enter the following command to enable the plugin in CEA:

```cea-config write --general:plugins remap_ville_plugin.RemapVillePlugin```

(If you have other plugins already, you can also add them to the general/plugins section in `cea.config`,
as a comma separated list)

Now you should be able to enter the following command to run the plugin:

```cea remap```