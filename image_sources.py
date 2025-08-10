# image_sources.py
import os
import requests
from urllib.parse import urljoin
from config import LIVE_INPUT_DIR, SERVER_FOLDER_URL, SERVER_LATEST_URL

# Ensure local input directory exists
os.makedirs(LIVE_INPUT_DIR, exist_ok=True)


def get_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to live_input/ and return file path.
    """
    file_path = os.path.join(LIVE_INPUT_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def list_local_images() -> list:
    """
    List all images in live_input/ directory.
    Returns list of file names (not full paths).
    """
    return [f for f in os.listdir(LIVE_INPUT_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]


def get_latest_image_from_server() -> str:
    """
    Download the latest image from the server and save to live_input/.
    Returns the saved file path.
    """
    response = requests.get(SERVER_LATEST_URL, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(LIVE_INPUT_DIR, "latest.jpg")
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return file_path
    else:
        raise Exception(f"Failed to fetch latest image: {response.status_code}")


def get_all_new_images_from_server() -> list:
    """
    Download all images from the server folder that are not already in live_input/.
    Returns list of saved file paths.
    """
    response = requests.get(SERVER_FOLDER_URL)
    if response.status_code != 200:
        raise Exception(f"Failed to list server folder: {response.status_code}")

    # Extract image URLs from HTML (simple parsing)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    image_urls = [urljoin(SERVER_FOLDER_URL, a['href'])
                  for a in soup.find_all('a', href=True)
                  if a['href'].lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    saved_files = []
    for img_url in image_urls:
        file_name = os.path.basename(img_url)
        local_path = os.path.join(LIVE_INPUT_DIR, file_name)
        if not os.path.exists(local_path):
            img_data = requests.get(img_url, stream=True)
            if img_data.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in img_data.iter_content(1024):
                        f.write(chunk)
                saved_files.append(local_path)

    return saved_files
