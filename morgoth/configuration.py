from morgoth.utils.file_utils import file_existing_and_readable
from morgoth.utils.package_data import get_path_of_user_dir
import yaml
import os

class MorgothConfig(object):

    def __init__(self):

        usr_path = get_path_of_user_dir()

        self._file_name = os.path.join(usr_path, 'morgoth_config.yml')
        
        if not file_existing_and_readable(self._file_name):

            print('morgoth config was not detected, creating a default one')



        with open(self._file_name, 'r') as f:
            self._configuration = yaml.load(f)


        
    def __getitem__(self, key):

        if key in self._configuration.keys():

            return self._configuration[key]

        else:

            raise ValueError("Configuration key %s does not exist in %s." % (key, self._filename))

    def __repr__(self):

        return yaml.dump(self._configuration, default_flow_style=False)
