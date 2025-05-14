# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import time
# from datetime import timedelta
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from sklearn.ensemble import IsolationForest
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler

# st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")

# WINDOW_SIZE = 14

# @st.cache_data
# def load_data():
#     df = pd.read_csv("netflix_users.csv", parse_dates=["Last_Login"])
#     df = df.sort_values("Last_Login")
#     return df

# df = load_data()

# target_date = pd.to_datetime("2025-03-08")
# date_range = pd.date_range(start=target_date, end=target_date, freq='D')

# bins = [0, 18, 25, 35, 50, 65, 120]
# labels = ['<18', '18-24', '25-34', '35-49', '50-64', '65+']
# df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

# np.random.seed(42)
# base_churn = 0.05
# max_churn = 0.154
# days_since = (target_date - df["Last_Login"]).dt.days.clip(lower=0)
# trend = base_churn + (days_since / days_since.max()) * (max_churn - base_churn)
# noise = np.random.normal(loc=0, scale=0.01, size=len(df))
# df["Churn_Rate"] = (trend + noise + np.sin(days_since / 15) * 0.02).clip(0.02, 0.25)

# spike_dates = ["2025-01-15", "2025-02-01", "2025-02-20"]
# spike_boost = 0.05
# for spike in spike_dates:
#     df.loc[df["Last_Login"] == pd.to_datetime(spike), "Churn_Rate"] += spike_boost

# df["Churn_Rate"] = df["Churn_Rate"].clip(0.02, 0.25)
# df["churn_flag"] = df["Churn_Rate"] > 0.15

# class LSTMModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=50, output_size=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.linear = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         out = self.linear(hn[-1])
#         return out

# def forecast_lstm(data, steps=14):
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(data.reshape(-1, 1))
#     X, y = [], []
#     for i in range(len(scaled) - WINDOW_SIZE):
#         X.append(scaled[i:i + WINDOW_SIZE])
#         y.append(scaled[i + WINDOW_SIZE])
#     X, y = np.array(X), np.array(y)

#     model = LSTMModel()
#     loss_fn = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     X_train = torch.tensor(X, dtype=torch.float32)
#     y_train = torch.tensor(y, dtype=torch.float32)

#     for epoch in range(100):
#         model.train()
#         output = model(X_train)
#         loss = loss_fn(output, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     preds = []
#     last_seq = X[-1]
#     for _ in range(steps):
#         with torch.no_grad():
#             pred = model(torch.tensor(last_seq.reshape(1, WINDOW_SIZE, 1), dtype=torch.float32))
#             preds.append(pred.item())
#             last_seq = np.roll(last_seq, -1)
#             last_seq[-1] = pred.item()

#     return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# current_date = target_date
# st.title("Netflix Time Series - Real-Time Dashboard")
# st.caption(f"Showing data up to: {current_date.strftime('%Y-%m-%d')}")
# df_current = df[df["Last_Login"] <= current_date].copy()

# login_counts = df_current.groupby("Last_Login").size().reset_index(name="Logins")

# try:
#     model_hw = ExponentialSmoothing(login_counts["Logins"], trend="add").fit()
#     future_dates = pd.date_range(current_date + timedelta(days=1), periods=14)
#     forecast_hw = model_hw.forecast(14)
#     forecast_hw_df = pd.DataFrame({"Last_Login": future_dates, "Logins": forecast_hw})
#     login_counts["Last_Login"] = pd.to_datetime(login_counts["Last_Login"]).dt.normalize()
#     forecast_hw_df["Last_Login"] = pd.to_datetime(forecast_hw_df["Last_Login"]).dt.normalize()
#     combined_hw = pd.concat([login_counts, forecast_hw_df]).drop_duplicates(subset="Last_Login").sort_values("Last_Login")
# except:
#     combined_hw = login_counts

# try:
#     login_array = login_counts["Logins"].values
#     forecast_lstm_vals = forecast_lstm(login_array)
#     lstm_dates = pd.date_range(current_date + timedelta(days=1), periods=14)
#     lstm_df = pd.DataFrame({"Last_Login": lstm_dates, "Logins": forecast_lstm_vals})
#     combined_lstm = pd.concat([login_counts, lstm_df])
# except:
#     combined_lstm = login_counts

# iso = IsolationForest(contamination=0.02)
# login_counts["anomaly"] = iso.fit_predict(login_counts[["Logins"]])
# login_counts["status"] = login_counts["anomaly"].map({1: "Normal", -1: "Anomaly"})

# st.metric("Total Watch Hours", f"{df_current['Watch_Time_Hours'].sum():,.0f} hrs")
# st.metric("Churn Rate", "15.4%")

# st.subheader("User Logins + ARIMA")
# st.plotly_chart(px.line(combined_hw, x="Last_Login", y="Logins"), use_container_width=True)

# st.subheader("User Logins + LSTM Forecast")
# st.plotly_chart(px.line(combined_lstm, x="Last_Login", y="Logins"), use_container_width=True)

# st.subheader("Churn Rate Over Time")
# churn_trend = df_current.groupby("Last_Login")["Churn_Rate"].mean().reset_index()
# st.plotly_chart(px.line(churn_trend, x="Last_Login", y="Churn_Rate"), use_container_width=True)

# st.subheader("Subscription Type Usage")
# sub_counts = df_current.groupby(["Last_Login", "Subscription_Type"]).size().reset_index(name="Count")
# st.plotly_chart(px.area(sub_counts, x="Last_Login", y="Count", color="Subscription_Type", groupnorm="fraction"), use_container_width=True)

# st.subheader("Average & Total Watch Time")
# avg_watch = df_current.groupby("Last_Login")["Watch_Time_Hours"].mean().reset_index()
# total_watch = df_current.groupby("Last_Login")["Watch_Time_Hours"].sum().reset_index()
# col1, col2 = st.columns(2)
# col1.plotly_chart(px.line(avg_watch, x="Last_Login", y="Watch_Time_Hours", title="Average Watch Time"), use_container_width=True)
# col2.plotly_chart(px.line(total_watch, x="Last_Login", y="Watch_Time_Hours", title="Total Watch Time"), use_container_width=True)

# st.subheader("Genre Popularity")
# genre_trend = df_current.groupby(["Last_Login", "Favorite_Genre"]).size().reset_index(name="Count")
# st.plotly_chart(px.area(genre_trend, x="Last_Login", y="Count", color="Favorite_Genre", groupnorm="fraction"), use_container_width=True)

# st.subheader("Age Group Distribution")
# age_dist = df_current["Age_Group"].value_counts().reset_index()
# age_dist.columns = ["Age Group", "Count"]
# st.plotly_chart(px.bar(age_dist, x="Age Group", y="Count", title="Age Group Distribution"), use_container_width=True)

# st.subheader("Anomaly Detection (Isolation Forest)")
# st.plotly_chart(
#     px.scatter(login_counts, x="Last_Login", y="Logins", color="status",
#                color_discrete_map={"Normal": "yellow", "Anomaly": "red"}),
#     use_container_width=True
# )



import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")

# Refresh every 1.5 seconds to speed things up
st_autorefresh(interval=1500, key="auto_refresh")

WINDOW_SIZE = 14

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_users.csv", parse_dates=["Last_Login"])
    df = df.sort_values("Last_Login")
    return df

df = load_data()
bins = [0, 18, 25, 35, 50, 65, 120]
labels = ['<18', '18-24', '25-34', '35-49', '50-64', '65+']
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

np.random.seed(42)
base_churn = 0.05
max_churn = 0.154
spike_dates = ["2025-01-15", "2025-02-01", "2025-02-20"]
spike_boost = 0.05

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=25, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.linear(hn[-1])
        return out

def forecast_lstm(data, steps=14):
    if len(data) < WINDOW_SIZE:
        return np.array([data.mean()] * steps)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - WINDOW_SIZE):
        X.append(scaled[i:i + WINDOW_SIZE])
        y.append(scaled[i + WINDOW_SIZE])
    X, y = np.array(X), np.array(y)

    model = LSTMModel()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    for epoch in range(25):
        model.train()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = []
    last_seq = X[-1]
    for _ in range(steps):
        with torch.no_grad():
            pred = model(torch.tensor(last_seq.reshape(1, WINDOW_SIZE, 1), dtype=torch.float32))
            preds.append(pred.item())
            last_seq = np.roll(last_seq, -1)
            last_seq[-1] = pred.item()

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# Simulate date forward
if "current_date" not in st.session_state:
    st.session_state.current_date = pd.to_datetime("2025-01-01")

current_date = st.session_state.current_date
end_date = pd.to_datetime("2025-03-08")

if current_date > end_date:
    st.session_state.current_date = pd.to_datetime("2025-01-01")
else:
    st.session_state.current_date += timedelta(days=1)

trend = base_churn + ((current_date - df["Last_Login"]).dt.days.clip(lower=0) / 60) * (max_churn - base_churn)
noise = np.random.normal(loc=0, scale=0.01, size=len(df))
df["Churn_Rate"] = (trend + noise + np.sin((current_date - df["Last_Login"]).dt.days / 15) * 0.02).clip(0.02, 0.25)
for spike in spike_dates:
    df.loc[df["Last_Login"] == pd.to_datetime(spike), "Churn_Rate"] += spike_boost
df["Churn_Rate"] = df["Churn_Rate"].clip(0.02, 0.25)
df["churn_flag"] = df["Churn_Rate"] > 0.15

df_current = df[df["Last_Login"] <= current_date].copy()
if df_current.empty:
    st.warning("Waiting for data to reach this date...")
    st.stop()

login_counts = df_current.groupby("Last_Login").size().reset_index(name="Logins")

try:
    model_hw = ExponentialSmoothing(login_counts["Logins"], trend="add").fit()
    future_dates = pd.date_range(current_date + timedelta(days=1), periods=14)
    forecast_hw = model_hw.forecast(14)
    forecast_hw_df = pd.DataFrame({"Last_Login": future_dates, "Logins": forecast_hw})
    combined_hw = pd.concat([login_counts, forecast_hw_df])
except:
    combined_hw = login_counts

try:
    login_array = login_counts["Logins"].values
    forecast_lstm_vals = forecast_lstm(login_array)
    lstm_dates = pd.date_range(current_date + timedelta(days=1), periods=14)
    lstm_df = pd.DataFrame({"Last_Login": lstm_dates, "Logins": forecast_lstm_vals})
    combined_lstm = pd.concat([login_counts, lstm_df])
except:
    combined_lstm = login_counts

iso = IsolationForest(contamination=0.02)
login_counts["anomaly"] = iso.fit_predict(login_counts[["Logins"]])
login_counts["status"] = login_counts["anomaly"].map({1: "Normal", -1: "Anomaly"})

st.title("Netflix Time Series - Dynamic Live Dashboard")
st.caption(f"Simulated date: {current_date.strftime('%Y-%m-%d')}")
st.metric("Total Watch Hours", f"{df_current['Watch_Time_Hours'].sum():,.0f} hrs")
st.metric("Churn Rate", f"{(df_current['churn_flag'].mean() * 100):.1f}%")

st.subheader("User Logins + ARIMA")
st.plotly_chart(px.line(combined_hw, x="Last_Login", y="Logins"), use_container_width=True)

st.subheader("User Logins + LSTM Forecast")
st.plotly_chart(px.line(combined_lstm, x="Last_Login", y="Logins"), use_container_width=True)

st.subheader("Churn Rate Over Time")
churn_trend = df_current.groupby("Last_Login")["Churn_Rate"].mean().reset_index()
st.plotly_chart(px.line(churn_trend, x="Last_Login", y="Churn_Rate"), use_container_width=True)

if "Subscription_Type" in df_current.columns:
    st.subheader("Subscription Type Usage")
    sub_counts = df_current.groupby(["Last_Login", "Subscription_Type"]).size().reset_index(name="Count")
    st.plotly_chart(px.area(sub_counts, x="Last_Login", y="Count", color="Subscription_Type", groupnorm="fraction"), use_container_width=True)

if "Watch_Time_Hours" in df_current.columns:
    st.subheader("Average & Total Watch Time")
    avg_watch = df_current.groupby("Last_Login")["Watch_Time_Hours"].mean().reset_index()
    total_watch = df_current.groupby("Last_Login")["Watch_Time_Hours"].sum().reset_index()
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.line(avg_watch, x="Last_Login", y="Watch_Time_Hours", title="Average Watch Time"), use_container_width=True)
    col2.plotly_chart(px.line(total_watch, x="Last_Login", y="Watch_Time_Hours", title="Total Watch Time"), use_container_width=True)

if "Favorite_Genre" in df_current.columns:
    st.subheader("Genre Popularity")
    genre_trend = df_current.groupby(["Last_Login", "Favorite_Genre"]).size().reset_index(name="Count")
    st.plotly_chart(px.area(genre_trend, x="Last_Login", y="Count", color="Favorite_Genre", groupnorm="fraction"), use_container_width=True)

if "Age_Group" in df_current.columns:
    st.subheader("Age Group Distribution")
    age_dist = df_current["Age_Group"].value_counts().reset_index()
    age_dist.columns = ["Age Group", "Count"]
    st.plotly_chart(px.bar(age_dist, x="Age Group", y="Count", title="Age Group Distribution"), use_container_width=True)

st.subheader("Anomaly Detection (Isolation Forest)")
st.plotly_chart(
    px.scatter(login_counts, x="Last_Login", y="Logins", color="status",
               color_discrete_map={"Normal": "yellow", "Anomaly": "red"}),
    use_container_width=True
)



