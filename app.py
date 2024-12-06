import pandas as pd
import streamlit as st
from parseFile import write_data, get_item_prices, db2df
import altair as alt

# set the layout to wide
st.set_page_config(layout="wide")


# make a cached function to get the data
@st.cache_data
def get_dataframe():
    return db2df(get_item_prices())


# create a file upload
uploaded_file = st.file_uploader('Upload your LUA file here.')

# To convert to a string based IO
if uploaded_file is not None:

    # get the file into a string
    stringio = uploaded_file.getvalue().decode("utf-8")

    # parse the string
    write_data(stringio)

    # clean the function cache
    # https://stackoverflow.com/a/77676594
    get_dataframe.clear()

# get the dataframe
__df = get_dataframe()
__names = list(__df['name'].unique())

# make an item selector
selection = set(st.multiselect('Choose the Items of Interest', options=__names, default=__names[0]))
c = alt.Chart(__df[__df['name'].isin(selection)]).mark_line(point=True).encode(x="date", y="price", color="name", tooltip=["date", "price", "name", 'Gold'])
st.altair_chart(c, use_container_width=True)
