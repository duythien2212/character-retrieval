import streamlit as st
from streamlit_image_select import image_select
import math
import os
from sklearn.metrics import auc
from extract_feature.arcface import ArcFace
from extract_feature.beit import BEiT
from extract_feature.clip import CLIP
from extract_feature.facenet import FaceNet
from merge.merge_by_counter import MergeByCounter
from merge.merge_by_distance import MergeByDistance
from merge.merge_by_rank import MergeByRank
from merge.merge_by_ranx import MergeByRanx
from retrieval.retrieval import RetrievalModel

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_PATH = "/content/drive/MyDrive/TRECVID/query2024"
SHOT_PATH = "/content/drive/MyDrive/TRECVID/shots"
PROJECT_PATH = "/content/drive/MyDrive/TRECVID"
DATABASE_PATH = os.path.join(PROJECT_PATH, "Group_Duy_Thang", "Features")
INDEX_PATH = os.path.join(PROJECT_PATH, "Group_Duy_Thang", "index")
GROUND_TRUTH_PATH = os.path.join(PROJECT_PATH, "Group_Duy_Thang", "ground-truth")

# Sidebar
with st.sidebar:
    st.write("Select input image query")

    # Select movie
    movies = os.listdir(QUERY_PATH)
    selected_movie = st.selectbox(
        "Select movie",
        movies,
    )
    movie_path = os.path.join(QUERY_PATH, selected_movie)

    # Select character
    characters = os.listdir(movie_path)
    selected_character = st.selectbox(
        "Select character",
        characters,
    )
    character_path = os.path.join(movie_path, selected_character)

    # Select face feature types
    face_models = [ArcFace, FaceNet]
    face_feature_types = ["ArcFace", "Facenet"]
    if "selected_face_model" not in st.session_state:
        st.session_state.selected_face_model = face_models[0]

    def update_selected_face_model():
        idx = face_feature_types.index(st.session_state.face_model)
        st.session_state.selected_face_model = face_models[idx]

    selected_feature_type = st.selectbox(
        "Select face feature types",
        face_feature_types,
        key="face_model",
        on_change=update_selected_face_model,
    )
    selected_face_model = st.session_state.selected_face_model

    # Select non-face feature types
    non_face_models = [CLIP, BEiT]
    non_face_feature_types = ["CLIP", "BEiT"]
    if "selected_non_face_model" not in st.session_state:
        st.session_state.selected_non_face_model = non_face_models[0]

    def update_selected_non_face_model():
        idx = non_face_feature_types.index(st.session_state.non_face_model)
        st.session_state.selected_non_face_model = non_face_models[idx]

    selected_non_face_feature_type = st.selectbox(
        "Select image feature types",
        non_face_feature_types,
        key="non_face_model",
        on_change=update_selected_non_face_model,
    )
    selected_non_face_model = st.session_state.selected_non_face_model

    # Select general option
    general_option = st.checkbox("Use general features (BEiT_s, CLIP_rn)", value=True)

    # Select remove type
    remove_types = ["Compare", "DBSCAN", ""]
    selected_remove_type = st.selectbox(
        "Select remove type",
        remove_types,
    )

    # Select merge type
    merge_models = [MergeByCounter, MergeByDistance, MergeByRank, MergeByRanx]
    merge_types = ["Counter", "Distance", "Rank", "Ranx"]
    if "selected_merge_model" not in st.session_state:
        st.session_state.selected_merge_model = merge_models[0]

    def update_selected_merge_model():
        idx = merge_types.index(st.session_state.merge_model)
        st.session_state.selected_merge_model = merge_models[idx]

    selected_merge_type = st.selectbox(
        "Select merge type",
        merge_types,
        key="merge_model",
        on_change=update_selected_merge_model,
    )
    selected_merge_model = st.session_state.selected_merge_model

    # Select top_k result
    top_k = 20
    top_k = st.number_input("Select top_k result", min_value=1, value=top_k, key="top_k")

    # Select column size
    q_col_size = 5
    q_col_size = st.number_input("Select query column size", min_value=1, value=q_col_size, key="q_col_size")

    # Select column size
    r_col_size = 3
    r_col_size = st.number_input("Select result column size", min_value=1, value=r_col_size, key="r_col_size")

# Main
st.title("Movie character retrieval")
st.write("Movie character retrieval based on image examples")

st.subheader("Image query")

query_image_names = os.listdir(character_path)
query_images = [os.path.join(character_path, img_name) for img_name in query_image_names]

grid = [st.columns(q_col_size) for i in range(math.ceil(len(query_images)/q_col_size))]

cols = st.columns(len(query_images))

is_selected = [True for i in range(len(query_images))]

for i in range(len(grid)):
    for j in range(q_col_size):
        with grid[i][j]:
            cur_idx = i*q_col_size + j
            if (cur_idx >= len(query_images)):
                continue
            st.image(query_images[cur_idx])
            is_selected[cur_idx] = st.checkbox(query_image_names[cur_idx], key=f"q-{cur_idx}")

# st.write(is_selected)

st.subheader("Results")

# Store result_images in session_state for persistence
if "result_images" not in st.session_state:
    st.session_state.result_images = [""] * top_k

result_images = st.session_state.result_images

# row_len = [col_size for i in range(math.ceil(len(images)/col_size))]
# st.write(row_len)

def search():
    char_db_path = os.path.join(INDEX_PATH, selected_movie)
    retrieval_model = RetrievalModel(
            char_db_path, 
            selected_face_model(),
            selected_non_face_model(),
            selected_merge_model(),
            None
          )
    retrieval_model.load_database(char_db_path, general_option, selected_remove_type)

    selected_queries = []
    for i in range(len(query_images)):
        if (is_selected[i]):
            selected_queries.append(query_images[i])
    
    results = retrieval_model.retrieval(selected_queries)

    return results

is_search = st.button("Search")

if is_search:
    result_images = search()
    result_images = result_images[:top_k]
    st.session_state.result_images = result_images  # Save to session_state

# Always use the session_state result_images for display
result_images = st.session_state.result_images
grid = [st.columns(r_col_size) for i in range(math.ceil(len(result_images)/r_col_size))]
result_movie_path = os.path.join(SHOT_PATH, selected_movie)
scene_names = [image.split('-')[0] + "-" + image.split('-')[1] for image in result_images]
result_scene_path = [os.path.join(result_movie_path, scene_names[i]) for i in range(len(scene_names))]
result_shot_path = [os.path.join(result_scene_path[i], result_images[i] + ".webm") for i in range(len(result_images))]


cols = st.columns(len(result_shot_path))
for i in range(len(grid)):
    for j in range(r_col_size):
        with grid[i][j]:
            cur_idx = i*r_col_size + j
            if (cur_idx >= len(result_shot_path)):
                continue
            st.video(result_shot_path[cur_idx])



