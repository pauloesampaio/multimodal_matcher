from utils import get_image, get_text_vector, get_image_vector, query_faiss
import faiss
from transformers import CLIPProcessor, CLIPModel
import torch as th
import numpy as np
import streamlit as st
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def get_index():
    return faiss.read_index("./data/image_index.faiss")


@st.cache_resource
def get_processor_and_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return processor, model


@st.cache_data
def load_products():
    base_url = os.getenv("BASE_URL")
    dataset_url = os.getenv("DATASET_URL")
    response = requests.get(dataset_url)
    all_products = json.loads(response.text)
    return all_products, base_url


st.set_page_config(layout="wide")
st.title("Multimodal product search")

image_index = get_index()
all_products, base_url = load_products()
query_vector = None

with st.spinner("Loading model..."):
    processor, model = get_processor_and_model()

search_type = st.radio("Search by", ["Image", "Text"], key="search_type")

if search_type == "Image":
    query_image = st.text_input("Enter image URL:")

if search_type == "Text":
    query_text = st.text_input("Enter search text:")

if st.button("Search"):
    if search_type == "Image":
        if query_image:
            try:
                image = get_image(query_image)
                query_vector = get_image_vector(image, processor, model)
            except Exception as e:
                st.error(f"Error processing image: {e}")
        else:
            st.warning("Please enter a valid text or image URL.")
    elif search_type == "Text":
        if query_text:
            try:
                query_vector = get_text_vector(query_text, processor, model)
            except Exception as e:
                st.error(f"Error processing text: {e}")
        else:
            st.warning("Please enter a valid text or image URL.")

    if query_vector is None:
        st.error("Failed to generate query vector. Please check your input.")
    else:
        top_5 = query_faiss(query_vector, image_index)

        st.subheader("Search Results")
        cols = st.columns(3)  # First row
        # Box 1: User input
        with cols[0]:
            if search_type == "Image" and query_image:
                st.markdown("# Image query")
                st.image(image, caption="Query Image", width=256)
            elif search_type == "Text" and query_text:
                st.markdown("# Text query")
                st.markdown(f"### {query_text}")

        # Boxes 2 and 3: Top 2 results
        for i, idx in enumerate(top_5[:2]):
            with cols[i + 1]:  # cols[1], cols[2]
                prod = all_products[idx]
                image_url = get_image(f"{base_url}{prod['images']}")
                st.image(image_url, caption=prod["title"], width=256)
                st.markdown(prod["details"])

        # Second row: 3 more results
        cols2 = st.columns(3)
        for i, idx in enumerate(top_5[2:]):
            with cols2[i]:
                prod = all_products[idx]
                image_url = get_image(f"{base_url}{prod['images']}")
                st.image(image_url, caption=prod["title"], width=256)
                st.markdown(prod["details"])
