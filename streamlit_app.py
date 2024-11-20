import streamlit as st
from streamlit_image_select import image_select
import math

# Sidebar
with st.sidebar:
    st.write("Select input image query")
    
    # Select movie
    movies = ["Movie 1", "Movie 2"]
    selected_movie = st.selectbox(
        "Select movie",
        movies,
    )

    # Select character 
    characters = ["Character 1", "Character 2", "Character 3"]
    selected_character = st.selectbox(
        "Select character",
        characters,
    )

    # Select face feature types
    face_feature_types = ["Facenet", "Arcface"]
    selected_feature_type = st.selectbox(
        "Select face feature types",
        face_feature_types,
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

query_images = ["images/cat1.jpeg", 
          "images/cat1.jpeg", 
          "images/cat1.jpeg", 
          "images/cat1.jpeg", 
          "images/cat1.jpeg", 
          "images/cat1.jpeg",
          "images/cat1.jpeg"]

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
            is_selected[cur_idx] = st.checkbox("Check me", key=f"q-{cur_idx}")

st.write(is_selected)

st.subheader("Results")

result_images = ["images/cat1.jpeg", 
          "images/cat1.jpeg", 
          "images/cat1.jpeg", 
          "images/cat1.jpeg", 
          "images/cat1.jpeg",
          "images/cat1.jpeg"]

# row_len = [col_size for i in range(math.ceil(len(images)/col_size))]
# st.write(row_len)



grid = [st.columns(col_size) for i in range(math.ceil(len(result_images)/col_size))]

cols = st.columns(len(result_images))
for i in range(len(grid)):
    for j in range(col_size):
        with grid[i][j]:
            cur_idx = i*col_size + j
            if (cur_idx >= len(result_images)):
                continue
            st.image(result_images[cur_idx])
            st.checkbox("Check me", key=f"r-{cur_idx}")



