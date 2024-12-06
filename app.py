import streamlit as st
from parseFile import write_data, get_item_prices, db2df, price2gold
import altair as alt
import io
import pandas as pd
import datetime
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns[:-1])
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if isinstance(df[column], pd.CategoricalDtype) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}, e.g. (?:CT|CP) for pressure and temperature",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

# set the layout to wide
st.set_page_config(layout="wide")


# make a cached function to get the data
@st.cache_data
def get_dataframe():
    return db2df(get_item_prices())


# IMPORTANT: Cache the conversion to prevent computation on every rerun
@st.cache_data
def convert_df(df, typed: str):
    if typed == 'text\csv':
        return df.to_csv().encode("utf-8")


    elif typed == 'application/vnd.ms-excel':
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Auctionator')
        return buffer
    else:
        raise ValueError('Unrecognized download type!')


# create a file upload
st.sidebar.header('File Upload')
uploaded_file = st.sidebar.file_uploader('Upload your LUA file here.')

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
__names = list(__df['Name'].unique())

# make an item selector
selection = set(st.multiselect('Choose the Items of Interest', options=__names, default=__names[0], ))

if selection:
    # select the items from the dataframe
    __selected_df = __df[__df['Name'].isin(selection)]
    __min_date = __selected_df['Date'].min().to_pydatetime()
    __max_date = __selected_df['Date'].max().to_pydatetime()

    # make a date selection
    date_range =  st.slider('Date Range', __min_date, __max_date+datetime.timedelta(days=1),
                            step=datetime.timedelta(days=1),
                            value=(__min_date, __max_date+datetime.timedelta(days=1)))

    # create the chard
    __selected_df = __selected_df[(pd.Timestamp(date_range[0]) <= __selected_df['Date']) &
                                  (__selected_df['Date'] <= pd.Timestamp(date_range[1]))]
    c = alt.Chart(__selected_df).mark_line(point=True).encode(x="Date", y="Price",
                                                              color=alt.Color("Name").legend(orient="bottom",
                                                                                             labelLimit=0,
                                                                                             symbolLimit=0,
                                                                                             columns=4),
                                                              tooltip=["Date", "Name", 'Gold'])
    st.altair_chart(c, use_container_width=True)

# create an overview per item
__grouped_df = __df.groupby('Name')
__grouped_df = __grouped_df['Price'].describe().join(
    __grouped_df[['Price']].apply(lambda x: [ele for ele in x.values]).to_frame('Prices'))

# style the dataframe
__cl_config = {"Prices": st.column_config.LineChartColumn("Prices")}
__style_format = dict()
for name in __grouped_df.columns[1:-1]:
    __style_format[name] = price2gold
__style_format['count'] = int

# make a checkbox whether to apply selection
apply_selection = st.checkbox('Apply Selection', value=False)

# display the dataframe
if apply_selection:
    __display_df = __grouped_df.loc[list(selection)]
else:
    __display_df = __grouped_df

# apply some filters
__display_df = filter_dataframe(__display_df)

# visualize the dataframe
st.write(f'Description of Price Distribution for {"the selected" if apply_selection else "all"}'
         f' Items (click columns to sort):')
st.dataframe(__display_df.style.format(__style_format), column_config=__cl_config)

# make a sidebar
with st.sidebar:

    st.sidebar.header('File Download')

    # make a selection for the sidebar
    download_type = st.selectbox('Choose the download type', options=['All', 'Selection'])

    # make the dataframe selection
    if download_type == 'All':
        __curr_df = __df
    elif download_type == 'Selection':
        __curr_df = __selected_df
    else:
        raise ValueError('Unspecified download!')

    # make a csv download button
    st.download_button(
        label="Download data as CSV",
        data=convert_df(__curr_df, 'text\csv'),
        file_name=f"{'selected' if download_type == 'Selection' else ''}_data.csv",
        mime="text/csv",
        use_container_width = True,
    )

    # Excel download button
    # MIME type: https://stackoverflow.com/a/1964182
    st.download_button(
        label="Download data as XLSX",
        data=convert_df(__curr_df, 'application/vnd.ms-excel'),
        file_name=f"{'selected' if download_type == 'Selection' else ''}_data.xlsx",
        mime="text/csv",
        use_container_width=True,
    )