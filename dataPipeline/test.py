from datasets import load_dataset
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import zipfile
import os

load_dotenv()

ds = load_dataset("InternScience/ChartX")

for split in ds:
    print(split)
    