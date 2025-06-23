from utils import get_image
import os
import requests
import json
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import torch as th
import numpy as np
import faiss

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
DATASET_URL = os.getenv("DATASET_URL")

print("Loading model")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


response = requests.get(DATASET_URL)
all_products = json.loads(response.text)

print("Loading data")
images = []
texts = []
base_url = "https://paulo-blog-media.s3.sa-east-1.amazonaws.com/genai/"
for product in all_products:
    image_url = f'{base_url}{product["images"]}'
    images.append(get_image(image_url))
    texts.append(f"{product['title']}. {product['details']}")

print("Encoding data")
model_input = processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
)

model.eval()
with th.no_grad():
    vectors = model(**model_input)

text_vectors = vectors["text_embeds"].numpy().astype("float32")
image_vectors = vectors["image_embeds"].numpy().astype("float32")
product_ids = np.array(range(len(image_vectors)))

print("Creating index")
faiss.normalize_L2(image_vectors)
image_index = faiss.IndexFlatIP(image_vectors.shape[1])
image_index.add(image_vectors)

if not os.path.exists("./data"):
    os.makedirs("./data")

faiss.write_index(image_index, "./data/image_index.faiss")
print("Image index saved to ./data/image_index.faiss")
