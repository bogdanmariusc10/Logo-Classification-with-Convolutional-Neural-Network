import pandas as pd
from datasets import load_dataset
from bs4 import BeautifulSoup
import requests
from PIL import Image
from PIL import UnidentifiedImageError
from io import BytesIO
import os
from urllib.parse import urlparse
import copy
from requests.exceptions import ConnectionError, Timeout, HTTPError
import cv2
import numpy as np
from tqdm import tqdm

# Define the labels
LABELS = {'Accessories': 0, 'Clothes': 1, 'Cosmetic': 2, 'Electronic': 3,
          'Food': 4, 'Institution': 5, 'Leisure': 6, 'Medical': 7,
          'Necessities': 8, 'Transportation': 9}

# Process the training and validation dataset
def get_dataset(directory):
    dataset = []
    class_count = [0 for label in LABELS]

    # Count total files for the progress bar
    total_files = sum(len(files) for _, _, files in os.walk(directory))

    with tqdm(total=total_files, desc="Processing Images", unit="img") as pbar:
        for root, _, files in os.walk(directory):
            category = os.path.basename(root)
            if category in LABELS:
                label_idx = LABELS[category]
                # Compute the one-hot-encoding
                one_hot_label = np.eye(len(LABELS))[label_idx]

                for file in files:
                    if file.lower().endswith('.jpg'):
                        # Convert the image to grayscale, resize it to 100x100 and append it with the one-hot-encoding
                        file_path = os.path.join(root, file)
                        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        image = cv2.resize(image, (100, 100))
                        dataset.append([np.array(image), one_hot_label])
                        class_count[label_idx] += 1
                        pbar.update(1)

    return dataset, class_count

# Process the prediction dataset
def get_logos():
    # Read the prediction dataset
    df = pd.read_parquet('logos.snappy.parquet')

    logo_list = []

    # Try to fetch the logos with Clearbite Logo API and website favicons
    for index, domain in enumerate(tqdm(df['domain'], desc="Processing Logos"), start=1):
        print(f"Processing {index}: {domain}")
        logo_url = f"https://logo.clearbit.com/{domain}"

        try:
            response = requests.get(logo_url, timeout=10)
            response.raise_for_status()

            logo_data = response.content
            img = Image.open(BytesIO(logo_data))

            # Convert to grayscale, resize and convert to NumPy array
            img = img.convert("L").resize((100, 100))
            img_array = np.array(img)

            logo_list.append([img_array, domain])
            continue

        # Treat the exceptions
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                print(f"Logo not found for {domain} (404). Trying favicon...")
            else:
                print(f"HTTP error for {domain}: {e}")

        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Network error for {domain}: {e}")

        # Try fetching the favicon if the logo fails
        favicon_url = f"https://{domain}/favicon.ico"

        try:
            favicon_response = requests.get(favicon_url, timeout=10)
            favicon_response.raise_for_status()

            logo_data = favicon_response.content
            img = Image.open(BytesIO(logo_data))

            # Convert to grayscale, resize and convert to NumPy array
            img = img.convert("L").resize((100, 100))
            img_array = np.array(img)

            logo_list.append([img_array, domain])

        # Treat exceptions
        except (requests.ConnectionError, requests.Timeout):
            print(f"Failed to fetch favicon for {domain}")
            logo_list.append(["FAIL", domain])

        except (requests.HTTPError, UnidentifiedImageError):
            print(f"Favicon for {domain} is not a valid image")
            logo_list.append(["FAIL", domain])

    return logo_list
