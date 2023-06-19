import json
import time

import requests

from morgoth.exceptions.custom_exceptions import (
    EmptyFileError,
    GRBNotFound,
    UnauthorizedRequest,
    UnexpectedStatusCode,
    UploadFailed,
)
from morgoth.utils.env import get_env_value

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")
base_url = get_env_value("MORGOTH_BASE_URL")
auth_token = get_env_value("MORGOTH_AUTH_TOKEN")

model_lookup = {"pl": "powerlaw", "cpl": "cutoff_powerlaw", "band": "band_function"}


def check_grb_on_website(grb_name):
    headers = {"Authorization": f"Token {auth_token}"}

    check_existing_url = f"{base_url}/api/check_grb_name/{grb_name}/"

    response = requests.get(url=check_existing_url, headers=headers, verify=False)

    # GRB not in DB
    if response.status_code == 204:
        return False

    # GRB already in DB
    elif response.status_code == 200:
        return True

    elif response.status_code == 401:
        raise UnauthorizedRequest("The authentication token is not valid")

    # This should not happen, but we will try to upload anyway
    else:
        return False


def create_report_from_result(result):
    if result["general"]["report_type"] == "trigdat":
        web_version = result["general"]["version"]
    else:
        web_version = (
            f"{result['general']['report_type']}_{result['general']['version']}"
        )

    report = {
        "name": result["general"]["grb_name"],
        "name_gcn": result["general"]["grb_name_gcn"],
        "hide_burst": False,
        "trigger_number": result["general"]["trigger_number"],
        "trigger_timestamp": result["general"]["trigger_timestamp"],
        "grb_params": [
            {
                "version": web_version,
                "model_type": model_lookup[result["fit_result"]["model"]],
                "trigger_number": result["general"]["trigger_number"],
                "trigger_timestamp": result["general"]["trigger_timestamp"],
                "data_timestamp": result["general"]["data_timestamp"],
                "localization_timestamp": result["general"]["localization_timestamp"],
                "balrog_ra": result["fit_result"]["ra"],
                "balrog_ra_err": result["fit_result"]["ra_err"],
                "balrog_dec": result["fit_result"]["dec"],
                "balrog_dec_err": result["fit_result"]["dec_err"],
                "swift_ra": result["general"]["swift"].get("ra", None)
                if result["general"]["swift"] is not None
                else None,
                "swift_dec": result["general"]["swift"].get("dec", None)
                if result["general"]["swift"] is not None
                else None,
                "spec_K": result["fit_result"]["spec_K"],
                "spec_K_err": result["fit_result"]["spec_K_err"],
                "spec_index": result["fit_result"]["spec_index"],
                "spec_index_err": result["fit_result"]["spec_index_err"],
                "spec_xc": result["fit_result"]["spec_xc"],
                "spec_xc_err": result["fit_result"]["spec_xc_err"],
                "sat_phi": result["fit_result"]["sat_phi"],
                "sat_theta": result["fit_result"]["sat_theta"],
                "spec_alpha": result["fit_result"]["spec_alpha"],
                "spec_alpha_err": result["fit_result"]["spec_alpha_err"],
                "spec_xp": result["fit_result"]["spec_xp"],
                "spec_xp_err": result["fit_result"]["spec_xp_err"],
                "spec_beta": result["fit_result"]["spec_beta"],
                "spec_beta_err": result["fit_result"]["spec_beta_err"],
                "bkg_neg_start": result["time_selection"]["bkg_neg_start"],
                "bkg_neg_stop": result["time_selection"]["bkg_neg_stop"],
                "bkg_pos_start": result["time_selection"]["bkg_pos_start"],
                "bkg_pos_stop": result["time_selection"]["bkg_pos_stop"],
                "active_time_start": result["time_selection"]["active_time_start"],
                "active_time_stop": result["time_selection"]["active_time_stop"],
                "used_detectors": ", ".join(
                    str(det_nr) for det_nr in result["time_selection"]["used_detectors"]
                ),
                "most_likely": result["general"]["most_likely"],
                "second_most_likely": result["general"]["second_most_likely"],
                "balrog_one_sig_err_circle": result["fit_result"][
                    "balrog_one_sig_err_circle"
                ],
                "balrog_two_sig_err_circle": result["fit_result"][
                    "balrog_two_sig_err_circle"
                ],
                "bright_sources": result["separation_values"]["bright_sources"],
                "SGRs": result["separation_values"]["SGRs"],
                "sun_separation": result["separation_values"]["sun_separation"],
                "sun_within_error": result["separation_values"]["sun_within_error"],
            }
        ],
    }
    return report


def upload_grb_report(grb_name, result, wait_time, max_time):
    headers = {
        "Authorization": "Token {}".format(auth_token),
        "Content-Type": "application/json",
    }
    upload_new_version = False
    grb_on_website = check_grb_on_website(grb_name)

    # Upload new version of report
    if grb_on_website:
        upload_new_version = True
        url = f"{base_url}/api/grbs/{grb_name}/params/"

    # Create GRB entry on website and upload report
    else:
        url = f"{base_url}/api/grbs/"

    report = create_report_from_result(result)

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    while flag:

        # try to download the file
        try:

            response = requests.post(
                url=url, data=json.dumps(report), headers=headers, verify=False
            )
            if response.status_code == 201:
                print("Uploaded new GRB")
                # kill the loop

                flag = False

            elif response.status_code == 206:
                print("Uploaded new GRB but notification went wrong")
                # kill the loop

                flag = False

            elif response.status_code == 401:
                raise UnauthorizedRequest("The authentication token is not valid")

            elif response.status_code == 409 and not upload_new_version:
                print("GRB already existing, try uploading new version")
                url = f"{base_url}/api/grbs/{grb_name}/params/"

            elif response.status_code == 409 and upload_new_version:
                print("################################################")
                print("The report for this version is already in the DB")
                print("################################################")
                flag = False

            else:
                raise UnexpectedStatusCode(
                    f"Request returned unexpected status {response.status_code} with message {response.text}"
                )

        except UnauthorizedRequest as e:
            raise e

        except Exception as e:

            print(e)

            # ok, we have not uploaded the grb yet

            # see if we should still wait for the upload

            if time_spent >= max_time:

                # we are out of time so give up

                raise UploadFailed("The max time has been exceeded!")

            else:

                # ok, let's sleep for a bit and then check again

                time.sleep(wait_time)

                # up date the time we have left

                time_spent += wait_time
        else:

            print(f"Response {response.status_code} at {url}: {response.text}")
    return report


def update_grb_report(grb_name, result, wait_time, max_time):
    headers = {
        "Authorization": "Token {}".format(auth_token),
        "Content-Type": "application/json",
    }

    # Update the GRB report on the website
    if check_grb_on_website(grb_name):
        url = f"{base_url}/'api/grbs/{grb_name}/params/"

    # Update of not possible if GRB is not already there
    else:
        raise GRBNotFound(
            f"Update of {grb_name} not possible, because it is not on Website"
        )

    report = create_report_from_result(result)

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    while flag:

        # try to download the file
        try:

            response = requests.put(
                url=url, data=json.dumps(report), headers=headers, verify=False
            )
            if response.status_code == 201:
                print("Updated GRB")
                # kill the loop

                flag = False

            elif response.status_code == 401:
                raise UnauthorizedRequest("The authentication token is not valid")

            else:
                raise UnexpectedStatusCode(
                    f"Request returned unexpected status {response.status_code} with message {response.text}"
                )

        except UnauthorizedRequest as e:
            raise e

        except Exception as e:

            print(e)

            # ok, we have not uploaded the grb yet

            # see if we should still wait for the upload

            if time_spent >= max_time:

                # we are out of time so give up

                raise UploadFailed("The max time has been exceeded!")

            else:

                # ok, let's sleep for a bit and then check again

                time.sleep(wait_time)

                # up date the time we have left

                time_spent += wait_time
        else:

            print(f"Response {response.status_code} at {url}: {response.text}")


def upload_plot(
    grb_name,
    report_type,
    plot_file,
    plot_type,
    version,
    wait_time,
    max_time,
    det_name="",
):
    headers = {
        "Authorization": "Token {}".format(auth_token),
    }

    web_version = version if report_type == "trigdat" else f"{report_type}_{version}"

    payload = {"plot_type": plot_type, "version": web_version, "det_name": det_name}

    # Update the GRB report on the website
    if check_grb_on_website(grb_name):
        url = f"{base_url}/api/grbs/{grb_name}/plot/"

    # Update of not possible if GRB is not already there
    else:
        raise GRBNotFound(
            f"Upload of plot for {grb_name} not possible, because GRB is missing"
        )

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    with open(plot_file, "rb") as file_:

        while flag:

            # try to download the file
            try:
                response = requests.post(
                    url=url,
                    data=payload,
                    headers=headers,
                    files={"file": file_},
                    verify=False,
                )
                if response.status_code == 201:
                    print("Uploaded new plot")
                    flag = False

                elif response.status_code == 401:
                    raise UnauthorizedRequest("The authentication token is not valid")

                elif response.status_code == 204:
                    raise EmptyFileError("The plot file is empty")

                elif response.status_code == 409:
                    print("####################################################")
                    print("The plot for this version is already in the Database")
                    print("####################################################")
                    flag = False

                else:
                    raise UnexpectedStatusCode(
                        f"Request returned unexpected status {response.status_code} with message {response.text}"
                    )

            except UnauthorizedRequest as e:
                raise e

            except EmptyFileError as e:
                raise e

            except Exception as e:

                print(e)

                # ok, we have not uploaded the grb yet

                # see if we should still wait for the upload

                if time_spent >= max_time:

                    # we are out of time so give up

                    raise UploadFailed("The max time has been exceeded!")

                else:

                    # ok, let's sleep for a bit and then check again

                    time.sleep(wait_time)

                    # up date the time we have left

                    time_spent += wait_time
            else:

                print(f"Response {response.status_code} at {url}: {response.text}")


def upload_datafile(
    grb_name, report_type, data_file, file_type, version, wait_time, max_time
):
    headers = {
        "Authorization": "Token {}".format(auth_token),
    }

    web_version = version if report_type == "trigdat" else f"{report_type}_{version}"

    payload = {"file_type": file_type, "version": web_version}

    # Update the GRB report on the website
    if check_grb_on_website(grb_name):
        url = f"{base_url}/api/grbs/{grb_name}/datafile/"

    # Update of not possible if GRB is not already there
    else:
        raise GRBNotFound(
            f"Upload of datafile for {grb_name} not possible, because GRB is missing"
        )

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    with open(data_file, "rb") as file_:

        while flag:

            # try to download the file
            try:
                response = requests.post(
                    url=url,
                    data=payload,
                    headers=headers,
                    files={"file": file_},
                    verify=False,
                )
                if response.status_code == 201:
                    print("Uploaded new data file")
                    flag = False

                elif response.status_code == 401:
                    raise UnauthorizedRequest("The authentication token is not valid")

                elif response.status_code == 204:
                    raise EmptyFileError("The datafile was empty")

                elif response.status_code == 409:
                    print("#########################################################")
                    print("The data file for this version is already in the Database")
                    print("#########################################################")
                    flag = False

                else:
                    raise UnexpectedStatusCode(
                        f"Request returned unexpected status {response.status_code} with message {response.text}"
                    )

            except UnauthorizedRequest as e:
                raise e

            except EmptyFileError as e:
                raise e

            except Exception as e:

                print(e)

                # ok, we have not uploaded the grb yet

                # see if we should still wait for the upload

                if time_spent >= max_time:

                    # we are out of time so give up

                    raise UploadFailed("The max time has been exceeded!")

                else:

                    # ok, let's sleep for a bit and then check again

                    time.sleep(wait_time)

                    # up date the time we have left

                    time_spent += wait_time
            else:

                print(f"Response {response.status_code} at {url}: {response.text}")
