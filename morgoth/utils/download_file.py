import os
import shutil
import time

import astropy.utils.data as astro_data

import morgoth.utils.file_utils as file_utils


def download_file(url, path="/tmp"):
    """
    Download a file to the given path
    """

    fname = url.split("/")[-1]
    f = astro_data.download_file(url)

    return f


class BackgroundDownload(object):
    def __init__(
        self, url, store_path=None, wait_time=60, max_time=60 * 60,
    ):
        """
        An worker to download objects in the background to avoid blocking the GCN
        listen function.

        If a bot is specfied, it will upload an the image with the bot.


        :param url: The URL to download the file
        :param bot: the optional bot
        :param description: the description for the bot's plot
        :param wait_time: the wait time interval for checking files
        :param max_time: the max time to wait for files
        :returns: 
        :rtype: 

        """

        self._wait_time = wait_time
        self._max_time = max_time

        self._url = url
        self._store_file = True

        # get the file name  at the end
        self._file_name = url.split("/")[-1]

        self._store_path = store_path

    def run(self):

        # set a flag to kill the job

        flag = True

        # the time spent waiting so far
        time_spent = 0  # seconds

        while flag:

            # try to download the file
            try:

                path = download_file(self._url)

                # create the directory

                file_utils.if_directory_not_existing_then_make(self._store_path)

                # move the file

                shutil.move(path, os.path.join(self._store_path, self._file_name))

                # kill the loop

                flag = False

            except:

                # ok, we have not found a file yet

                # see if we should still wait for the file

                if time_spent >= self._max_time:

                    # we are out of time so give up

                    flag = False

                else:

                    # ok, let's sleep for a bit and then check again

                    time.sleep(self._wait_time)

                    # up date the time we have left

                    time_spent += self._wait_time

        return path
