from datetime import datetime
import os
import requests
import json
import time

from morgoth.exceptions.custom_exceptions import GRBNotFound

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")
base_url = os.environ.get("MORGOTH_BASE_URL")
auth_token = os.environ.get("MORGOTH_AUTH_TOKEN")


def check_grb_on_website(grb_name):
    headers = {

        'Authorization': f"Token {auth_token}"

    }

    check_existing_url = f"{base_url}/api/check_grb_name/{grb_name}/"

    response = requests.get(url=check_existing_url, headers=headers, verify=False)

    # GRB not in DB
    if response.status_code == 204:
        return True

    # GRB already in DB
    elif response.status_code == 200:
        return False

    # This should not happen, but we will try to upload anyway
    else:
        return False


def upload_grb_report(grb_name, report):
    headers = {

        'Authorization': 'Token {}'.format(auth_token),
        'Content-Type': 'application/json',

    }

    grb_on_website = check_grb_on_website(grb_name)

    # Upload new version of report
    if grb_on_website:
        url = f"{base_url}/'api/grbs/{grb_name}/params/"

    # Create GRB entry on website and upload report
    else:
        url = f"{base_url}/api/grbs/"

    send = False
    while not send:
        try:

            response = requests.post(url=url, data=json.dumps(report), headers=headers, verify=False)
            if response.status_code == 201:
                print('Uploaded new GRB')
                send = True

        except:

            print("Connection timed out!")
            send = False
        else:

            print('{}: {}'.format(response.status_code, response.text))


def update_grb_report(grb_name, report):
    headers = {
        'Authorization': 'Token {}'.format(auth_token),
        'Content-Type': 'application/json',
    }

    # Update the GRB report on the website
    if check_grb_on_website(grb_name):
        url = f"{base_url}/'api/grbs/{grb_name}/params/"

    # Update of not possible if GRB is not already there
    else:
        raise GRBNotFound(f"Update of {grb_name} not possible, because it is not on Website")

    send = False
    while not send:
        try:

            response = requests.put(url=url, data=json.dumps(report), headers=headers, verify=False)
            if response.status_code == 201:
                print('Uploaded new GRB')
                send = True

        except:

            print("Connection timed out!")
            send = False
        else:

            print('{}: {}'.format(response.status_code, response.text))


def upload_plot(grb_name, report_type, plot_file, plot_type, version, det_name=None):
    headers = {

        'Authorization': 'Token {}'.format(auth_token),

    }

    web_version = version if report_type == 'trigdat' else f"{report_type}_{version}"

    if det_name is not None:
        payload = {

            'plot_type': plot_type,
            'version': web_version,
            'det_name': det_name

        }
    else:
        payload = {

            'plot_type': plot_type,
            'version': web_version

        }

    # Update the GRB report on the website
    if check_grb_on_website(grb_name):
        url = f"{base_url}/'api/grbs/{grb_name}/plot/"

    # Update of not possible if GRB is not already there
    else:
        raise GRBNotFound(f"Upload of plot for {grb_name} not possible, because GRB is missing")

    send = False
    with open(plot_file, 'rb') as file_:

        while not send:
            try:

                response = requests.post(url=url, data=payload, headers=headers,
                                         files={"file": file_}, verify=False)
                send = True

            except:

                print("Connection timed out!")
                time.sleep(1)
                send = False
            else:

                print('{}: {}'.format(response.status_code, response.text))


def upload_datafile(grb_name, report_type, data_file, file_type, version):
    headers = {

        'Authorization': 'Token {}'.format(auth_token),

    }

    web_version = version if report_type == 'trigdat' else f"{report_type}_{version}"

    payload = {

        'file_type': file_type,
        'version': web_version

    }

    # Update the GRB report on the website
    if check_grb_on_website(grb_name):
        url = f"{base_url}/'api/grbs/{grb_name}/datafile/"

    # Update of not possible if GRB is not already there
    else:
        raise GRBNotFound(f"Upload of plot for {grb_name} not possible, because GRB is missing")

    send = False
    with open(data_file, 'rb') as file_:

        while not send:

            try:

                response = requests.post(url=url, data=payload, headers=headers,
                                         files={"file": file_}, verify=False)
                send = True

            except:

                print("Connection timed out!")
                send = False
            else:

                print('{}: {}'.format(response.status_code, response.text))
