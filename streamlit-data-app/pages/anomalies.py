import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import plotly.graph_objects as go  
import numpy as np  
import pydeck as pdk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from helpers import query_database


@st.cache_data(ttl=30)  # only re-query if it's been 30 seconds
def get_anomalies(anomaly_id):
    if not anomaly_id:
        return pd.DataFrame()
    return query_database(
        """
        select 
            first_dimension, second_dimension, third_dimension, anomaly_scores, anomaly_tag
        from 
            hackathon.were_just_here_for_the_snacks.feature_may_last3wk_btl 
        """
        # where anomaly_id = '{anomaly_id}'
    )


def _3d_scatter_vs_tag(data):

    range1 = data["first_dimension"].max() - data["first_dimension"].min()
    range2 = data["second_dimension"].max() - data["second_dimension"].min()
    range3 = data["third_dimension"].max() - data["third_dimension"].min()

    if range1 > range2 and range1 > range3:
        x = data["first_dimension"]
    elif range2 > range1 and range2 > range3:
        x = data["second_dimension"]
    elif range3 > range1 and range3 > range2:
        x = data["third_dimension"]

    if (range1 > range2 and range1 < range3) or (range1 < range2 and range1 > range3):
        y = data["first_dimension"]
    elif (range2 > range1 and range2 < range3) or (range2 < range1 and range2 > range3):
        y = data["second_dimension"]
    elif (range3 > range1 and range3 < range2) or (range3 < range1 and range3 > range2):
        y = data["third_dimension"]

    if (range1 < range2 and range1 < range3):
        z = data["first_dimension"]
    elif (range2 < range1 and range2 < range3):
        z = data["second_dimension"]
    elif (range3 < range1 and range3 < range2):
        z = data["third_dimension"]

    cluster_data = pd.DataFrame()
    cluster_data["first_dimension"] = x
    cluster_data["second_dimension"] = y
    cluster_data["third_dimension"] = z
    cluster_data["anomaly_scores"] = 1 - ((data["anomaly_scores"] - data["anomaly_scores"].min()) / (data["anomaly_scores"].max() - data["anomaly_scores"].min()))

    # Create a 3D scatter plot
    fig = px.scatter_3d(
        cluster_data, 
        x="first_dimension", y="second_dimension", z="third_dimension", 
        color="anomaly_scores", 
        color_continuous_scale=["green", "yellow", "red"],
        width=1000, 
        height=500,
    )
    fig.update_layout(
        title={
            'text': "3D Representation of High-Dimensional Data",
            'font': {
                'size': 24  # Change this value to adjust the title size
            }
        }
    )
    st.plotly_chart(fig)


def _3d_surface_vs_score(data):

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[['first_dimension', 'second_dimension', 'third_dimension']])

    #kmeans = KMeans(n_clusters=2)
    #kmeans.fit_transform(scaled)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(scaled)

    x_grid, y_grid = np.meshgrid(
        np.linspace(data_pca[:, 0].min(), data_pca[:, 0].max(), 50),
        np.linspace(data_pca[:, 1].min(), data_pca[:, 1].max(), 50),
    )
    z_grid = griddata(
        (data_pca[:, 0], data_pca[:, 1]),
        1-((data["anomaly_scores"] - data["anomaly_scores"].min()) /(data["anomaly_scores"].max() - data["anomaly_scores"].min())),
        (x_grid, y_grid),
        method="cubic",
        fill_value=1
    )

    # Create a 3D scatter plot
    # scatter_df = pd.DataFrame()
    # scatter_df["first_dimension"] = data_pca[:, 0]
    # scatter_df["second_dimension"] = data_pca[:, 1]
    # scatter_df["anomaly_score"] = data["anomaly_score"]
    # fig = px.scatter_3d(scatter_df, x="first_dimension", y="second_dimension", z="anomaly_score", color="anomaly_tag", title="3D Clustered Data")
    # st.plotly_chart(fig)

    # inliers = data["anomaly_tag"] == 0
    # outliers = data["anomaly_tag"] == 1
    # scatter_df = pd.DataFrame()
    # scatter_df["first_dimension"] = data_pca[inliers, 0]
    # scatter_df["second_dimension"] = data_pca[inliers, 1]
    # scatter_df["third_dimension"] = data["anomaly_score"][inliers]
    # fig = px.scatter_3d(scatter_df, x="first_dimension", y="second_dimension", z="anomaly_score", color="anomaly_tag", title="3D Clustered Data")
    # st.plotly_chart(fig)

    # Create a 3D plot  
    fig = go.Figure(data=[go.Surface(z=z_grid, x=x_grid, y=y_grid, colorscale="rdylgn_r")])
    fig.update_layout(
        title={
            'text': "Anomaly Landscape",
            'font': {
                'size': 24  # Change this value to adjust the title size
            }
        }, 
        autosize=False, 
        width=1000, 
        height=500,  
        # margin=dict(l=65, r=50, b=65, t=90)
    )  
    st.plotly_chart(fig)


def show_anomalies():

    st.title("Anomalies")

    st.markdown(
        """
        <style>
        .center-table {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    selected_anomaly = st.selectbox("Choose an Anomaly", ["Anomaly #1 (2025-05-06)", "Anomaly #2 (2025-06-01)", "Anomaly #3 (2025-06-02)"], index=None)
    data = get_anomalies(selected_anomaly)

    if not data.empty:
        anomaly = data[['anomaly_tag', 'anomaly_scores']].groupby("anomaly_tag", as_index=True).count().rename(columns={"anomaly_scores": "Count"}).sort_index()
        anomaly.index = ['Not Anomalous', 'Anomalous']
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.dataframe(anomaly, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        _3d_scatter_vs_tag(data)
        _3d_surface_vs_score(data)
        
