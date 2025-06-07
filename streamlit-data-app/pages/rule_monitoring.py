import streamlit as st
import pandas as pd
from helpers import query_database, st_normal
from plotly.subplots import make_subplots
import plotly.graph_objects as go


@st.cache_data(ttl=30)  # only re-query if it's been 30 seconds
def get_rule_performance_data():
    return query_database(
        """
        select 
            rule_num, gen_vol, frd_vol, frd_val, fpr, dw_bus_dt
        from 
            hackathon.were_just_here_for_the_snacks.final_rules_perf_daily_badfpr
        """ # final_rules_perf_daily
        # where anomaly_id = '{anomaly_id}'
    )


def highlight_rows(row):
    return ['background-color: rgba(255, 182, 193, 0.5)' if row.rule_num == 'rule_3' else ('background-color: rgba(255, 255, 224, 0.5)' if row.rule_num == 'rule_0' else 'background-color: rgba(144, 238, 144, 0.5)') for _ in row]


def show_rule_monitoring():

    st.markdown("<h2 style='text-align: center; color: black;'>Overall Ruleset Performance</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>Recent Week vs. Previous Week</h3>", unsafe_allow_html=True)

    rule_df = get_rule_performance_data()

    aggregated_ruleset_performance = rule_df.groupby(
        "dw_bus_dt", as_index=False
    ).agg(
        gen_vol=("gen_vol", "sum"),
        frd_vol=("frd_vol", "sum"),
        frd_val=("frd_val", "sum"),
    ).sort_values(
        by='dw_bus_dt'
    )

    aggregated_ruleset_performance['dw_bus_dt'] = pd.to_datetime(aggregated_ruleset_performance['dw_bus_dt'])
    aggregated_ruleset_performance.set_index('dw_bus_dt', inplace=True)
    
    rolling_window = aggregated_ruleset_performance.rolling(
        window='30D'
    ).agg({
        'gen_vol': 'sum',
        'frd_vol': 'sum',
        'frd_val': 'sum',
    }).reset_index()
    rolling_window['fpr'] = round(rolling_window['frd_vol'] / rolling_window['gen_vol'], 2)

    current_performance = rolling_window.iloc[-1].fillna(0)
    old_performance = rolling_window.iloc[-1-14].fillna(0)

    with st_normal():

        a, b = st.columns(2)
        c, d = st.columns(2)

        a.metric("# Genuines", int(current_performance['gen_vol']), int(current_performance['gen_vol']-old_performance['gen_vol']), delta_color='inverse')
        b.metric("# Frauds", int(current_performance['frd_vol']), int(current_performance['frd_vol']-old_performance['frd_vol']))
        c.metric("Fraud Value", f"£ {round(current_performance['frd_val']/1_000_000, 2)} M", f"£ {current_performance['frd_val']-old_performance['frd_val']}", delta_color='inverse')
        d.metric("FPR", f"{current_performance['fpr']}%", f"{round(current_performance['fpr']-old_performance['fpr'], 2)}%", delta_color='inverse')

    aggregated_rules_performance = rule_df.groupby(
        ["rule_num"], as_index=False
    ).agg(
        gen_vol=("gen_vol", "sum"),
        frd_vol=("frd_vol", "sum"),
        frd_val=("frd_val", "sum"),
    ).sort_values(
        by=["rule_num"]
    )
    aggregated_rules_performance['fpr'] = round(aggregated_rules_performance['frd_vol'] / aggregated_rules_performance['gen_vol'], 2)
    aggregated_rules_performance = aggregated_rules_performance.fillna(0)

    st.markdown("</br><h2 style='text-align: center; color: black;'>Individual Rule Performance</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>Recent Week vs. Previous Week</h3>", unsafe_allow_html=True)
    styled_df = aggregated_rules_performance.style.apply(highlight_rows, axis=1)
    selected_row = st.dataframe(styled_df, use_container_width=True, on_select="rerun", selection_mode="single-row")

    if len(selected_row.selection.rows) > 0:
        
        selected_options = aggregated_rules_performance["rule_num"].iloc[selected_row.selection.rows].tolist()

        rule_df = rule_df[
            rule_df["rule_num"].isin(selected_options)
        ].groupby(
            "dw_bus_dt", as_index=False
        ).agg(
            gen_vol=("gen_vol", "sum"),
            frd_vol=("frd_vol", "sum"),
            frd_val=("frd_val", "sum"),
        ).sort_values(
            by='dw_bus_dt'
        )
        rule_df['fpr'] = round(rule_df['frd_vol'] / rule_df['gen_vol'], 2)
        rule_df = rule_df.fillna(0)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Number of Genuine", "Number of Frauds", "Total Fraud Value", "FPR"))
        fig.add_trace(go.Scatter(x=rule_df["dw_bus_dt"], y=rule_df["gen_vol"], mode='lines', name='gen_vol', legendgroup='group1'), row=1, col=1)
        fig.add_trace(go.Scatter(x=rule_df["dw_bus_dt"], y=rule_df["frd_vol"], mode='lines', name='frd_vol', legendgroup='group2'), row=2, col=1)
        fig.add_trace(go.Scatter(x=rule_df["dw_bus_dt"], y=rule_df["frd_val"], mode='lines', name='frd_val', legendgroup='group3'), row=3, col=1)
        fig.add_trace(go.Scatter(x=rule_df["dw_bus_dt"], y=rule_df["fpr"], mode='lines', name='fpr', legendgroup='group4'), row=4, col=1)
        fig.update_yaxes(range=[0, None], row=1, col=1)
        fig.update_yaxes(range=[0, None], row=2, col=1)
        fig.update_yaxes(range=[0, None], type='log', row=3, col=1)
        fig.update_yaxes(range=[0, None], row=4, col=1)
        fig.update_layout(
            height=1000, 
            width=1000,
            showlegend=False,
            title={
                'text': "Rule Firings Over Time",
                'font': {
                    'size': 24  # Change this value to adjust the title size
                },
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Anchor the title at the center
            }
        )
        st.plotly_chart(fig)


