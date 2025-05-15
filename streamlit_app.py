import streamlit as st
from streamlit_image_select import image_select
import math
import os
import torch
from deepface import DeepFace
from PIL import Image
import pandas as pd
import numpy as np
import faiss
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import auc
from pipeline.retrieval import Retrieval
from merge.merge_result import merge_result

query_path = "./query"
shot_path = "/content/drive/MyDrive/TRECVID/shots"

# Sidebar
with st.sidebar:
    st.write("Select input image query")

    # Select movie
    movies = os.listdir(query_path)
    selected_movie = st.selectbox(
        "Select movie",
        movies,
    )
    movie_path = os.path.join(query_path, selected_movie)

    # Select character
    characters = os.listdir(movie_path)
    selected_character = st.selectbox(
        "Select character",
        characters,
    )
    character_path = os.path.join(movie_path, selected_character)

    # Select face feature types
    face_feature_types = ["Facenet", "Arcface"]
    selected_feature_type = st.selectbox(
        "Select face feature types",
        face_feature_types,
    )

    # Select face feature types
    non_face_feature_types = ["CLIP", "BEiT"]
    selected_non_face_feature_type = st.selectbox(
        "Select image feature types",
        non_face_feature_types,
    )

    # Select general option
    general_option = st.checkbox("Use general features (BEiT_s, CLIP_rn)", value=True)

    # Select remove type
    remove_types = ["Compare", "DBSCAN", ""]
    selected_remove_type = st.selectbox(
        "Select remove type",
        remove_types,
    )

    # Select merge type
    merge_types = ["no", "Distance", "Counter", "Ranx"]
    selected_merge_type = st.selectbox(
        "Select merge type",
        merge_types,
    )

    # Select top_k result
    top_k = 5
    top_k = st.number_input("Select top_k result", min_value=1, value=top_k, key="top_k")

    # Select column size
    col_size = 5
    col_size = st.number_input("Select column size", min_value=1, value=col_size, key="col_size")

# Main
st.title("Movie character retrieval")
st.write("Movie character retrieval based on image examples")

st.subheader("Image query")

query_image_names = os.listdir(character_path)
query_images = [os.path.join(character_path, img_name) for img_name in query_image_names]

# img_names = [img.split('/')[-1].split('.')[0] for img in query_images]

# row_len = [col_size for i in range(math.ceil(len(images)/col_size))]
# st.write(row_len)

grid = [st.columns(col_size) for i in range(math.ceil(len(query_images)/col_size))]

cols = st.columns(len(query_images))

is_selected = [False for i in range(len(query_images))]

for i in range(len(grid)):
    for j in range(col_size):
        with grid[i][j]:
            cur_idx = i*col_size + j
            if (cur_idx >= len(query_images)):
                continue
            st.image(query_images[cur_idx])
            is_selected[cur_idx] = st.checkbox(query_image_names[cur_idx], key=f"q-{cur_idx}")

st.write(is_selected)

st.subheader("Results")

result_images = []

# row_len = [col_size for i in range(math.ceil(len(images)/col_size))]
# st.write(row_len)

def search():
    retrieval = Retrieval(query_images, selected_feature_type, selected_non_face_feature_type,
                         remove_type=selected_remove_type, general=general_option, film=selected_movie)
    D_with_face, I_with_face, results_with_face = retrieval.search_with_face(top_k)
    D_without_face, I_without_face, results_without_face = retrieval.search_without_face(top_k)

    results = merge_result(results_with_face, results_without_face, D_with_face, D_without_face, type=selected_merge_type)

    return results

if st.button("Search"):
    result_images = search()
    # st.write("Why hello there")
    grid = [st.columns(col_size) for i in range(math.ceil(len(result_images)/col_size))]
    result_movie_path = os.path.join(shot_path, selected_movie)
    scene_names = [image.split('-')[0] + "-" + image.split('-')[1] for image in result_images]
    result_scene_path = [os.path.join(result_movie_path[i], scene_names[i]) for i  in range(len(scene_names))]
    result_shot_path = [os.path.join(result_scene_path[i], result_images[i] + ".webm") for i  in range(len(result_images))]


    cols = st.columns(len(result_shot_path))
    for i in range(len(grid)):
        for j in range(col_size):
            with grid[i][j]:
                cur_idx = i*col_size + j
                if (cur_idx >= len(result_shot_path)):
                    continue
                st.video(result_shot_path[cur_idx])



