import luigi
import os
import shutil
os.environ['GBM_TRIGGER_DATA_DIR'] = './'
import lxml.etree
import time
from morgoth.trigger import parse_trigger_file_and_write, OpenGBMFile
from morgoth.reports import CreateAllPages
from morgoth.downloaders import DownloadTrigdat
from morgoth.utils.package_data import get_path_of_data_file


from morgoth.configuration import morgoth_config

for i in range(3):

    v = f"v0{i}"
    morgoth_config['download']['trigdat'][v]['max_time'] = 10


# def test_parse_trigger():

#     ff = get_path_of_data_file("gbm_flt.xml")
#     with open(ff, "r") as f:
#         root = lxml.etree.parse(f)

#     grb = parse_trigger_file_and_write(root)

    

    
#     assert luigi.build(
#         [OpenGBMFile(grb=grb)], local_scheduler=True
#     )


# def test_download_trigdat():

#     ff = get_path_of_data_file("gbm_flt.xml")
#     with open(ff, "r") as f:
#         root = lxml.etree.parse(f)

#     grb = parse_trigger_file_and_write(root)

    
#     assert luigi.build(
#         [DownloadTrigdat(grb_name=grb, version='v01')], local_scheduler=False,
#         scheduler_host='localhost'
#     )


def test_pipeline():

    ff = get_path_of_data_file("gbm_flt.xml")
    with open(ff, "r") as f:
        root = lxml.etree.parse(f)

    grb = parse_trigger_file_and_write(root)

    time.sleep(5)
    assert luigi.build(
        [CreateAllPages(grb_name=grb)], local_scheduler=False,
        scheduler_host='localhost', workers=4
    )
    # luigi.build(
    #     [CreateAllPages(grb_name=grb)], local_scheduler=True,
    #     workers=6, no_lock=False
    # )

    
    shutil.rmtree(grb)
