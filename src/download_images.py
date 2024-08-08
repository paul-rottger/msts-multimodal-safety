import os
import requests
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def download_images(image_url, image_name, save_directory):
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:

        # add a custom user agent to the request -> required by some websites
        headers = {'User-Agent': 'PaulRottger/0.0 (paul.rottger@unibocconi.it)'}

        response = requests.get(image_url, headers=headers)
        response.raise_for_status()  # Check if the request was successful

        image_path = os.path.join(save_directory, image_name+".jpg")

        with open(image_path, 'wb') as file:
            file.write(response.content)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {image_name} from {image_url}: {e}")


# SET PARAMETERS
df_path = "data/unsafe_images.csv"
df_name_col = "unsafe_image_id"
df_url_col = "unsafe_image_url"
save_directory = "./data/unsafe_images"

# load data
image_df = pd.read_csv(df_path)

# sample 10 for testing
image_df = image_df.sample(10, random_state=42)

# download images
image_df.progress_apply(lambda row: download_images(image_url = row[df_url_col], image_name = row[df_name_col], save_directory = save_directory), axis=1)