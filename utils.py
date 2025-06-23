import requests
from io import BytesIO
import numpy as np
import torch as th
from PIL import Image
import faiss


def get_image(url):
    """
    Given a url, return the PIL image
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        if 200 <= response.status_code < 300:
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print(f"Request failed with code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def get_text_vector(query_text, processor, model):
    """
    Get feature vector of a text using the processor and model
    """
    query_input = processor(text=query_text, return_tensors="pt")
    with th.no_grad():
        query_vector = model.get_text_features(**query_input)
    return query_vector.numpy().astype("float32")


def get_image_vector(query_image, processor, model):
    """
    Get feature vector of a text using the processor and model
    """
    query_input = processor(images=query_image, return_tensors="pt")
    with th.no_grad():
        query_vector = model.get_image_features(**query_input)
    return query_vector.numpy().astype("float32")


def query_faiss(query_vector, index, top_k=5):
    """
    Receives an image or a text (defined in the data type), embeds
    accordingly using a multimodal model, and uses the embeded vector
    to query a pre-computed index. Returns the top_k closest results.
    """

    faiss.normalize_L2(query_vector)
    _, indices = index.search(query_vector, top_k)
    return indices[0]
