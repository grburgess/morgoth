import pytest
import lxml.etree
import os
os.environ["GBM_TRIGGER_DATA_DIR"] = "./"

from morgoth.utils.package_data import get_path_of_data_file
from morgoth.trigger import parse_trigger_file_and_write
@pytest.fixture(scope="session")
def payload1():

    ff = get_path_of_data_file("gbm_flt.xml")

    with open(ff,'rb') as f:

        payload = f.read()


    return payload
    





@pytest.fixture(scope="session")
def payload2():

    ff = get_path_of_data_file("gbm_flt2.xml")

    with open(ff,'rb') as f:

        payload = f.read()


    return payload




@pytest.fixture(scope="session")
def root1():

    ff = get_path_of_data_file("gbm_flt.xml")
    with open(ff, "r") as f:
        root = lxml.etree.parse(f)

    return root


@pytest.fixture(scope="session")
def root2():

    ff = get_path_of_data_file("gbm_flt2.xml")
    with open(ff, "r") as f:
        root = lxml.etree.parse(f)

    return root


        

@pytest.fixture(scope="session")
def grb1(root1, payload1):

    grb = parse_trigger_file_and_write(root1, payload1)

    return grb

@pytest.fixture(scope="session")
def grb2(root2, payload2):

    grb = parse_trigger_file_and_write(root2, payload2)

    return grb
