from databricks import sql
import streamlit as st

colours = {
    "barclays_blue": "#14375A",
    "barclays_medium_blue": "#204D76",
    "barclays_light_blue": "#00AEEF",
    "barclays_white": "#FFFFFF",
    "barclays_grey": "#A0A0A0",
}


def query_database(query):  
    with sql.connect(
        server_hostname="<add>",
        http_path="<add>",
        access_token="<add>"
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
        

def st_normal():
    _, col, _ = st.columns([1, 4, 1])
    return col
