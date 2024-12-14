import logging

import streamlit as st
import altair as alt
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def get_remote_ip() -> [str | None]:
    """Get remote ip."""
    # https://github.com/NginxProxyManager/nginx-proxy-manager/issues/674#issuecomment-717459734 for Nginx Proxy Manager
    # https://docs.streamlit.io/develop/api-reference/utilities/st.context for headers
    # print(list(st.context.headers.keys()))
    return st.context.headers.get('X-Real-Ip', 'IP Unknown (not forwarded)')


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.user_ip = get_remote_ip()
        return super().filter(record)


def init_logging():
    # Make sure to instantiate the logger only once
    # otherwise, it will create a StreamHandler at every run
    # and duplicate the messages
    # https://stackoverflow.com/a/75437429

    # create a custom logger
    logger = logging.getLogger("auctionator")
    if logger.handlers:  # logger is already setup, don't setup again
        return
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # in the formatter, use the variable "user_ip"
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s [user_ip=%(user_ip)s] - %(message)s")

    # set the ip handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.addFilter(ContextFilter())
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # set the file handler
    handler = logging.FileHandler("auctionator-debug.log")
    handler.setLevel(logging.INFO)
    handler.addFilter(ContextFilter())
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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
