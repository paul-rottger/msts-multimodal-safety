import os
import requests
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from PIL import Image

tqdm.pandas()

def download_images(image_url, image_name, save_directory, overwrite=False):

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:

        # add a custom user agent to the request -> required by some websites
        headers = {'User-Agent': 'PaulRottger/0.0 (paul.rottger@unibocconi.it)'}

        response = requests.get(image_url, headers=headers)
        response.raise_for_status()  # Check if the request was successful

        image_path = os.path.join(save_directory, image_name+".jpg")

        if os.path.exists(image_path) and not overwrite:
            return
        img = Image.open(BytesIO(response.content))

        # remove alpha channel
        if img.mode == 'RGBA' or img.mode == "LA":
            img = img.convert('RGB')

        img.save(image_path)

    #except requests.exceptions.RequestException as e:
    except Exception as e:
        print(f"Failed to download {image_name} from {image_url}: {e}")


# SET PARAMETERS
df_path = "data/unsafe_images_v0.2.csv"
df_name_col = "unsafe_image_id"
df_url_col = "unsafe_image_url"
save_directory = "./data/unsafe_images"

# load data
image_df = pd.read_csv(df_path)

# sample 10 for testing
# image_df = image_df.sample(10, random_state=42)

# download images
image_df.progress_apply(lambda row: download_images(image_url = row[df_url_col], image_name = row[df_name_col], save_directory = save_directory), axis=1)
