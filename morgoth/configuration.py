from morgoth.utils.file_utils import file_existing_and_readable, if_directory_not_existing_then_make
from morgoth.utils.package_data import get_path_of_user_dir, get_path_of_data_file
import yaml
import os
import shutil


class MorgothConfig(object):
    def __init__(self):

        usr_path = get_path_of_user_dir()

        self._file_name = os.path.join(usr_path, "morgoth_config.yml")

        # create the usr path if it is not there
        
        if_directory_not_existing_then_make(usr_path)
        
        # copy the default config to the usr directory if there is not
        # one
        if not file_existing_and_readable(self._file_name):

            print("morgoth config was not detected, creating a default one")

            default_file = get_path_of_data_file("morgoth_config.yml")

            shutil.copyfile(default_file, self._file_name)

        # now load the configuration
        
        with open(self._file_name, "r") as f:
            self._configuration = yaml.load(f)

        # it is currently not safe and can easily be corrupted

            
    def __getitem__(self, key):

        if key in self._configuration.keys():

            return self._configuration[key]

        else:

            raise ValueError(
                "Configuration key %s does not exist in %s." % (key, self._filename)
            )

    def __repr__(self):

        return yaml.dump(self._configuration, default_flow_style=False)



morgith_config = MorgothConfig()


__all__=['morgith_config']
