import streamlit as st

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
    top_k = st.number_input("Select top_k result", min_value=1, value=top_k)


# Main
st.title("Movie character retrieval")
st.write("Movie character retrieval based on image examples")

img1 = st.image("https://static.streamlit.io/examples/cat.jpg", width=100),
    
options = [
    img1,
    img1,
    img1
    ]
selection = st.segmented_control(
    "Directions", options, selection_mode="multi"
)
st.markdown(f"Your selected options: {selection}.")


