import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from helpers import query_database, st_normal


@st.cache_data(ttl=30)  # only re-query if it's been 30 seconds
def get_rule_performance_data():
    old_rules = query_database(
        """
        select 
            rule_num, gen_vol, frd_vol, frd_val, fpr, dw_bus_dt
        from 
            hackathon.were_just_here_for_the_snacks.final_rules_perf_daily_badfpr
        """ # final_rules_perf_daily
        # where anomaly_id = '{anomaly_id}'
    )
    new_rules = query_database(
        """
        select 
            rule_num, gen_vol, frd_vol, frd_val, fpr, dw_bus_dt
        from 
            hackathon.were_just_here_for_the_snacks.final_rules_perf_daily_postop
        """ # final_rules_perf_daily
        # where anomaly_id = '{anomaly_id}'
    )
    return old_rules, new_rules


def show_rule_optimisation():
    st.title("Ruleset Optimisations")

    old_rules_df, new_rules_df = get_rule_performance_data()

    selected_anomaly = st.selectbox("Select Ruleset Iteration", ['2025-05-31', '2025-06-01', '2025-06-02', '2025-06-03'], index=None)

    if selected_anomaly:

        old_rules_df = old_rules_df.groupby(
            "dw_bus_dt", as_index=False
        ).agg(
            gen_vol=("gen_vol", "sum"),
            frd_vol=("frd_vol", "sum"),
            frd_val=("frd_val", "sum"),
        ).sort_values(
            by='dw_bus_dt'
        )
        old_rules_df['fpr'] = old_rules_df['frd_vol'] / old_rules_df['gen_vol']

        new_rules_df = new_rules_df.groupby(
            "dw_bus_dt", as_index=False
        ).agg(
            gen_vol=("gen_vol", "sum"),
            frd_vol=("frd_vol", "sum"),
            frd_val=("frd_val", "sum"),
        ).sort_values(
            by='dw_bus_dt'
        )
        new_rules_df['fpr'] = new_rules_df['frd_vol'] / new_rules_df['gen_vol']

        # joined_df = pd.merge(old_rules_df, new_rules_df, on='dw_bus_dt', suffixes=('_old', '_new'))

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Number of Genuine", "Number of Frauds", "Total Fraud Value", "FPR"))
        fig.add_trace(go.Scatter(x=old_rules_df["dw_bus_dt"], y=old_rules_df["gen_vol"], mode='lines', name='current_ruleset', legendgroup='group1'), row=1, col=1)
        fig.add_trace(go.Scatter(x=new_rules_df["dw_bus_dt"], y=new_rules_df["gen_vol"], mode='lines', name='new_ruleset', legendgroup='group1'), row=1, col=1)

        fig.add_trace(go.Scatter(x=old_rules_df["dw_bus_dt"], y=old_rules_df["frd_vol"], mode='lines', name='current_ruleset', legendgroup='group2'), row=2, col=1)
        fig.add_trace(go.Scatter(x=new_rules_df["dw_bus_dt"], y=new_rules_df["frd_vol"], mode='lines', name='new_ruleset', legendgroup='group2'), row=2, col=1)

        fig.add_trace(go.Scatter(x=old_rules_df["dw_bus_dt"], y=old_rules_df["frd_val"], mode='lines', name='current_ruleset', legendgroup='group3'), row=3, col=1)
        fig.add_trace(go.Scatter(x=new_rules_df["dw_bus_dt"], y=new_rules_df["frd_val"], mode='lines', name='new_ruleset', legendgroup='group3'), row=3, col=1)

        fig.add_trace(go.Scatter(x=old_rules_df["dw_bus_dt"], y=old_rules_df["fpr"], mode='lines', name='current_ruleset', legendgroup='group4'), row=4, col=1)
        fig.add_trace(go.Scatter(x=new_rules_df["dw_bus_dt"], y=new_rules_df["fpr"], mode='lines', name='new_ruleset', legendgroup='group4'), row=4, col=1)

        fig.update_yaxes(range=[0, None], row=1, col=1)
        fig.update_yaxes(range=[0, None], row=2, col=1)
        fig.update_yaxes(range=[0, None], type='log', row=3, col=1)
        fig.update_yaxes(range=[0, None], row=4, col=1)

        fig.update_layout(height=1000, width=1000, title={
                'text': "Rule Firings Over Time",
                'font': {
                    'size': 24  # Change this value to adjust the title size
                },
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Anchor the title at the center
            })
        st.plotly_chart(fig)