import os
#from databricks import sql
#from databricks.sdk.core import Config
import streamlit as st
#import pandas as pd
#from helpers import colours
#from clustering import show_clustering_page
#from rules import show_rules_page
#from PIL import Image
from streamlit_navigation_bar import st_navbar
import pages as pg

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# st.set_page_config(layout="wide")

# st.markdown("""
#     <style>
#         h1 {
#             font-size: 50px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# col1, col2 = st.columns([6, 1])

# with col1:
#     st.title("Show me the snacks bruh")

# with col2:
#     st.image(Image.open("static/logo.png"), width=100)

# layout="wide", 
st.set_page_config(initial_sidebar_state="collapsed")

pages = ["Home", "Anomalies", "Rule Suggestions", "Rule Monitoring", "Ruleset Optimisation"]
logo_path = "static/logo.svg"
styles = {
    "nav": {
        "background-color": "royalblue",
        "justify-content": "center",
    },
    "img": {
        "padding-right": "14px",
    },
    "div": {
        "max-width": "50rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "white",
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "14px",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}
page = st_navbar(
    pages,
    logo_path=logo_path,
    styles=styles,
    options=options,
)

functions = {
    "Home": pg.show_home,
    "Anomalies": pg.show_anomalies,
    "Rule Suggestions": pg.show_rule_suggestions,
    "Rule Monitoring": pg.show_rule_monitoring,
    "Ruleset Optimisation": pg.show_rule_optimisation,
}
go_to = functions.get(page)
if go_to:
    go_to()
