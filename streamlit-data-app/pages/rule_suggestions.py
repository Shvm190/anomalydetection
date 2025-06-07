import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from helpers import query_database, st_normal


@st.cache_data(ttl=30)  # only re-query if it's been 30 seconds
def get_rule_performance_data():
    return query_database(
        """
        select 
            rule_num, gen_vol, frd_vol, frd_val, fpr, dw_bus_dt
        from 
            hackathon.were_just_here_for_the_snacks.final_rules_perf_daily_postop
        """ # final_rules_perf_daily
        # where anomaly_id = '{anomaly_id}'
    )


def show_rule_suggestions():
    st.title("Rule Suggestions")

    df = get_rule_performance_data()
    df['Anomaly ID'] = 'Anomaly #1 (2025-05-06)'

    selected_anomaly = st.selectbox("Select Anomaly", ['Anomaly #1 (2025-05-06)', "Anomaly #2 (2025-06-01)", "Anomaly #3 (2025-06-02)"], index=None)

    # Filter dataframe based on selected rule
    anomaly_df = df[df["Anomaly ID"] == selected_anomaly]

    anomaly_aggregated_options_df = anomaly_df.groupby(
        "rule_num", as_index=False
    ).agg(
        {"gen_vol": "sum", "frd_vol": "sum", "frd_val": "sum", "dw_bus_dt": "count"}
    ).rename(
        columns={"dw_bus_dt": "num_days_tested"}
    )

    # Display filtered dataframe
    if selected_anomaly:
        selected_row = st.dataframe(anomaly_aggregated_options_df, on_select="rerun", selection_mode="single-row", use_container_width=True)

        if len(selected_row.selection.rows) > 0:
            # Get the selected row data

            selected_options = anomaly_aggregated_options_df["rule_num"].iloc[selected_row.selection.rows].tolist()

            rule_df = anomaly_df[anomaly_df["rule_num"].isin(selected_options)].sort_values("dw_bus_dt")

            # Display image
            # image_filename = f"{rule_df["cluster"]}_{rule_df["option"]}.png"
            # st.title(f"{anomaly_df['Anomaly ID'].iloc[0]}_{anomaly_df['rule_num'].iloc[0]}")
            with st.expander("Rule Diagram", expanded=False, icon=":material/lan:"):
                image_filename = "static/tree.png"
                st.image(image_filename, caption=image_filename, width=600)

            with st.expander("Rule Backtest Performance", expanded=False, icon=":material/lan:"):
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Number of Genuine", "Number of Frauds", "Total Fraud Value", "FPR"))
                fig.add_trace(go.Scatter(x=rule_df["dw_bus_dt"], y=rule_df["gen_vol"], mode='lines', name='gen_vol', legendgroup='group1'), row=1, col=1)
                fig.add_trace(go.Scatter(x=rule_df["dw_bus_dt"], y=rule_df["frd_vol"], mode='lines', name='frd_vol', legendgroup='group2'), row=2, col=1)
                fig.add_trace(go.Scatter(x=rule_df["dw_bus_dt"], y=rule_df["frd_val"], mode='lines', name='frd_val', legendgroup='group3'), row=3, col=1)
                fig.add_trace(go.Scatter(x=rule_df["dw_bus_dt"], y=rule_df["fpr"], mode='lines', name='fpr', legendgroup='group4'), row=4, col=1)
                fig.update_yaxes(range=[0, None], row=1, col=1)
                fig.update_yaxes(range=[0, None], row=2, col=1)
                fig.update_yaxes(range=[0, None], type='log', row=3, col=1)
                fig.update_yaxes(range=[0, None], row=4, col=1)
                fig.update_layout(height=1000, width=1000, showlegend=False,
                                  title={
                'text': "Rule Firings Over Time",
                'font': {
                    'size': 24  # Change this value to adjust the title size
                },
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Anchor the title at the center
            })
                #fig = px.line(rule_df, x="date", y="num_fraud", title="Line Plot of Table Data")
                st.plotly_chart(fig)
