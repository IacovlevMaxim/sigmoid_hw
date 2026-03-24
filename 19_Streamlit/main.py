import streamlit as st

st.set_page_config( page_title="Wine Dataset", page_icon="🍷" )

pages = [
    st.Page("welcome.py", title="Welcome", icon="👋"),
    st.Page("dataset_info.py", title="Dataset Information", icon=":material/info:"),
    st.Page("dataset_import.py", title="Importing the Dataset", icon=":material/download:"),
    st.Page("training.py", title="Train the Model", icon=":material/exercise:")
]

pg = st.navigation(pages, position='sidebar')
pg.run()