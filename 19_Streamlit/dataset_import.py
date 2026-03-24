import streamlit as st
import pandas as pd

st.title("Wine Dataset Page 🍷")

st.header("Importing the Dataset")

importing = "wine_df = pd.read_csv(\"wine-quality-white-and-red.csv\")"
exec(importing)
st.code(importing)

sample = "wine_df.sample(5)"
st.code(sample)
st.write(eval(sample))