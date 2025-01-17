import os
import sys
import requests
import tqdm


def get_existing_files(api_token, persistent_id, dataverse_url):
    """
    Retrieves a list of existing files in the Dataverse dataset.

    Args:
        api_token (str): Dataverse API token.
        persistent_id (str): Persistent ID of the dataset.
        dataverse_url (str): Dataverse base URL.

    Returns:
        set: A set of filenames already in the dataset.
    """
    response = requests.get(
        f"{dataverse_url}/api/datasets/:persistentId/versions/:latest/files",
        headers={"X-Dataverse-key": api_token},
        params={"persistentId": persistent_id},
    )
    if response.status_code != 200:
        print(f"Failed to retrieve dataset files: {response.text}")
        sys.exit(1)

    files = response.json().get("data", [])
    return {file["dataFile"]["filename"] for file in files if "dataFile" in file}


def upload_files_to_dataverse(directory, api_token, persistent_id, dataverse_url):
    """
    Uploads all files in the given directory to a Dataverse dataset, skipping existing files.

    Args:
        directory (str): Path to the directory containing files to upload.
        api_token (str): Dataverse API token.
        persistent_id (str): Persistent ID of the dataset.
        dataverse_url (str): Dataverse base URL.
    """
    if not os.path.isdir(directory):
        print(
            f"Error: The specified directory '{directory}' does not exist or is not a directory."
        )
        sys.exit(1)

    existing_files = get_existing_files(api_token, persistent_id, dataverse_url)

    for filename in tqdm.tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if (
                filename in existing_files
                or filename.endswith(".xmp")
                or filename.endswith("tif")
            ):
                print(f"Skipping {filename}: already exists in the dataset.")
                continue

            print(f"Uploading {filename}...")
            with open(filepath, "rb") as file:
                response = requests.post(
                    f"{dataverse_url}/api/datasets/:persistentId/add",
                    headers={"X-Dataverse-key": api_token},
                    files={"file": (filename, file)},
                    params={"persistentId": persistent_id},
                )
            if response.status_code == 200:
                print(f"Uploaded {filename} successfully.")
            else:
                print(f"Failed to upload {filename}: {response.text}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upload_to_dataverse.py <directory>")
        sys.exit(1)

    # Arguments
    directory = sys.argv[1]
    # read api_token from file cfg/api_token
    with open("cfg/dataverse_api_token") as f:
        api_token = f.read().strip()
    # api_token = "your-api-token"
    persistent_id = "doi:10.14428/DVN/DEQCIM"
    dataverse_url = "https://dataverse.uclouvain.be"

    # Upload files
    upload_files_to_dataverse(directory, api_token, persistent_id, dataverse_url)
