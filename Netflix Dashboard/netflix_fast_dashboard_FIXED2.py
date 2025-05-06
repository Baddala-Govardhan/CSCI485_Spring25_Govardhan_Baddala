
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

df = load_data()
min_date = df["Last_Login"].min()
max_date = df["Last_Login"].max()
date_range = pd.date_range(start=min_date, end=max_date)

# Age group
bins = [0, 18, 25, 35, 50, 65, 120]
labels = ['<18', '18-24', '25-34', '35-49', '50-64', '65+']
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

placeholder = st.empty()

for current_date in date_range:
    with placeholder.container():
        st.title("🎬 Netflix AI-Powered Real-Time Dashboard")
        st.caption(f"📅 Data up to: {current_date.strftime('%Y-%m-%d')}")
        df_current = df[df["Last_Login"] <= current_date].copy()

        if len(df_current) < 20:
            st.warning("Waiting for more data to begin predictions...")
            time.sleep(0.5)
            continue

        df_current["churn_flag"] = (current_date - df_current["Last_Login"]).dt.days > 30
        login_counts = df_current.groupby("Last_Login").size().reset_index(name="Logins")
        churn_trend = df_current.groupby("Last_Login")["churn_flag"].mean().reset_index()
        sub_counts = df_current.groupby(["Last_Login", "Subscription_Type"]).size().reset_index(name="Count")
        watch_avg = df_current.groupby("Last_Login")["Watch_Time_Hours"].mean().reset_index()
        watch_total = df_current.groupby("Last_Login")["Watch_Time_Hours"].sum().reset_index()
        genre_trend = df_current.groupby(["Last_Login", "Favorite_Genre"]).size().reset_index(name="Count")
        age_dist = df_current["Age_Group"].value_counts().reset_index()
        age_dist.columns = ["Age Group", "Count"]

        # Holt-Winters Forecast
        login_forecast_plot = login_counts.copy()
        login_forecast_plot.set_index("Last_Login", inplace=True)
        try:
            model = ExponentialSmoothing(login_forecast_plot["Logins"], trend="add", seasonal=None).fit()
            future_dates = pd.date_range(current_date + timedelta(days=1), periods=14)
            forecast_values = model.forecast(14)
            forecast_hw = pd.DataFrame({"Last_Login": future_dates, "Logins": forecast_values})
            login_combined = pd.concat([login_counts, forecast_hw])
        except Exception as e:
            forecast_hw = pd.DataFrame()
            login_combined = login_counts.copy()  # fallback to original
            print("⚠️ Forecast failed, showing only raw data:", e)
    

        # Anomaly detection on login
        iso_model = IsolationForest(contamination=0.02, random_state=42)
        login_counts["anomaly"] = iso_model.fit_predict(login_counts[["Logins"]])
        anomalies = login_counts[login_counts["anomaly"] == -1]

        # Exportable datasets
        st.download_button("📥 Export Login Data", data=login_counts.to_csv(index=False), file_name="logins.csv")
        st.download_button("📥 Export Watch Time Data", data=watch_avg.to_csv(index=False), file_name="watch_time.csv")
        st.download_button("📥 Export Anomaly Report", data=anomalies.to_csv(index=False), file_name="anomalies.csv")

        st.metric("📊 Total Watch Hours", f"{df_current['Watch_Time_Hours'].sum():,.0f} hrs")
        st.metric("🧠 Churn Rate", f"{df_current['churn_flag'].mean():.2%}")

        st.subheader("Login Trend + Forecasts")
        fig_login = px.line(login_combined, x=login_combined.index, y="Logins", title="Logins + Holt-Winters Forecast")
        st.plotly_chart(fig_login, use_container_width=True)

        fig_anomaly = px.scatter(login_counts, x="Last_Login", y="Logins", color=login_counts["anomaly"].map({1: "Normal", -1: "Anomaly"}),
                                 title="Anomaly Detection on Logins")
        st.plotly_chart(fig_anomaly, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_sub = px.area(sub_counts, x="Last_Login", y="Count", color="Subscription_Type",
                              title="Logins by Subscription Type", groupnorm="fraction")
            st.plotly_chart(fig_sub, use_container_width=True)
        with col2:
            fig_watch = px.line(watch_avg, x="Last_Login", y="Watch_Time_Hours",
                                title="Avg Watch Time Over Time")
            st.plotly_chart(fig_watch, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig_churn = px.line(churn_trend, x="Last_Login", y="churn_flag", title="Churn Rate Over Time")
            st.plotly_chart(fig_churn, use_container_width=True)
        with col4:
            fig_total = px.line(watch_total, x="Last_Login", y="Watch_Time_Hours", title="Total Watch Hours Over Time")
            st.plotly_chart(fig_total, use_container_width=True)

        fig_genre = px.area(genre_trend, x="Last_Login", y="Count", color="Favorite_Genre",
                            title="Genre Popularity Over Time", groupnorm="fraction")
        st.plotly_chart(fig_genre, use_container_width=True)

        st.subheader("Age Group Distribution")
        fig_age = px.bar(age_dist.sort_values("Age Group"), x="Age Group", y="Count", title="Age Group Distribution")
        st.plotly_chart(fig_age, use_container_width=True)

        time.sleep(0.5)
