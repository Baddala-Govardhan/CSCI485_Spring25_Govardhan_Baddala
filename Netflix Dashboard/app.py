import os

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_CONFIG_FILE"] = ".streamlit/config.toml"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Netflix AI Real-Time Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_users.csv", parse_dates=["Last_Login"])
    df = df.sort_values("Last_Login")
    return df

# Remainder of the logic should go here...