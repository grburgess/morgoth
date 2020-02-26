from morgoth.utils.file_utils import file_existing_and_readable, if_directory_not_existing_then_make
from morgoth.utils.package_data import get_path_of_user_dir, get_path_of_data_file
import yaml
import os
import shutil


class MorgothConfig(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            #    print('Creating the object')
            cls._instance = super(MorgothConfig, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance

    def __init__(self):

        usr_path = get_path_of_user_dir()

        self._filename = os.path.join(usr_path, "morgoth_config.yml")

        # create the usr path if it is not there
        
        if_directory_not_existing_then_make(usr_path)
        
        # copy the default config to the usr directory if there is not
        # one
        if not file_existing_and_readable(self._filename):

            print("morgoth config was not detected, creating a default one")

            default_file = get_path_of_data_file("morgoth_config.yml")

            shutil.copyfile(default_file, self._filename)

        # now load the configuration
        
        with open(self._filename, "r") as f:
            self._configuration = yaml.load(f,Loader=yaml.SafeLoader)

        # it is currently not safe and can easily be corrupted

            
    def __getitem__(self, key):

        if key in self._configuration:

            return self._configuration[key]

        else:

            raise ValueError(
                "Configuration key %s does not exist in %s." % (key, self._filename)
            )

    def __repr__(self):

        return yaml.dump(self._configuration, default_flow_style=False)



morgoth_config = MorgothConfig()


__all__=['morgoth_config']
