import streamlit as st
import os
import glob
import json
import pandas as pd
import pygwalker as pyg
import altair as alt


# from pie_matrix import render_pie_matrix
# from learning_curves import render_learning_curves, render_all_learning_curves_bottom
# from leaderboard import render_leaderboard
# from heatmap import render_heatmap
# from heat_properties import render_heat_properties
# from medaillienspiegel import render_medaillienspiegel
# from lit_values_ui import render_lit_values_interface

# --- Main App Initialization ---
st.set_page_config(layout="wide")
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title("Nablachem Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    if os.path.exists("results_dataframe.parquet"):
        df = pd.read_parquet("results_dataframe.parquet")
        return df
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data found in results_dataframe.parquet. Please run `dashboard_manager.py` to generate the dataframe.")
else:
    # Set up Pages/Tabs
    tab1, tab2 = st.tabs(["Learning Curves & Properties", "Data Exploration"])

    with tab1:
        st.header("Combined Learning Curves")
        
        # Dropdowns
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.selectbox("Select Dataset", options=sorted(df['dataset'].unique()))
        with col2:
            prop_opts = sorted(df[df['dataset'] == dataset]['column_name'].unique())
            prop = st.selectbox("Select Property", options=prop_opts)
            
        # Filter logic: all representations each just test MAE and only best kernel (use validation MAE)
        filtered = df[(df['dataset'] == dataset) & (df['column_name'] == prop)]
        
        if not filtered.empty:
            best_kernels = []
            reps = filtered['representation'].unique()
            for rep in reps:
                rep_data = filtered[filtered['representation'] == rep]
                # find kernel with minimum validation_mae at max ntrain
                max_ntrain = rep_data['ntrain'].max()
                max_ntrain_data = rep_data[rep_data['ntrain'] == max_ntrain]
                if not max_ntrain_data.empty:
                    best_k = max_ntrain_data.loc[max_ntrain_data['validation_mae'].idxmin()]['kernel']
                    best_kernels.append({'representation': rep, 'kernel': best_k})
            
            plot_data = []
            for bk in best_kernels:
                b_rep = bk['representation']
                b_kern = bk['kernel']
                b_df = filtered[(filtered['representation'] == b_rep) & (filtered['kernel'] == b_kern)]
                
                b_df_agg = b_df.groupby('ntrain')['test_mae'].mean().reset_index()
                b_df_agg['representation'] = b_rep
                b_df_agg['kernel'] = b_kern
                b_df_agg['rep_kernel'] = f"{b_rep} ({b_kern})"
                plot_data.append(b_df_agg)
                
            if plot_data:
                plot_df = pd.concat(plot_data)
                plot_df = plot_df[plot_df['ntrain'] >= 128]
                
                chart = alt.Chart(plot_df).mark_line(point=True).encode(
                    x=alt.X('ntrain:Q', scale=alt.Scale(type='log', base=2), title='Training Size (log2)'),
                    y=alt.Y('test_mae:Q', scale=alt.Scale(type='log', base=10), title='Test MAE (log10)'),
                    color=alt.Color('rep_kernel:N', title='Representation (Kernel)'),
                    tooltip=['representation', 'kernel', 'ntrain', 'test_mae']
                ).properties(
                    width='container',
                    height=400
                ).interactive()
                st.altair_chart(chart, width='stretch')
            else:
                st.info("No data available for plotting.")
            
        st.markdown(" Plot of test MAE vs trainings steps.")
        st.markdown("Only best kernel is shown for each representation, based on the validation MAE at max trainig sizes.")
        
        st.markdown("---")
        st.header("Properties Too Difficult to Learn")
        st.write("Ranked by Nullmodel error / model error at 10k (lowest percentage is 1st place)")
        
        diff_props = []
        for p in df['column_name'].unique():
            p_df = df[df['column_name'] == p]
            # Get best representation & kernel for this property at max ntrain
            max_n = p_df['ntrain'].max()
            if max_n < 10000:
                continue
                
            best_model_run = p_df[p_df['ntrain'] == max_n].sort_values('test_mae').iloc[0]
            model_error = best_model_run['test_mae']
            
            # Find null model error (ntrain = 2 for the same rep/kernel, or overall max error)
            min_n = p_df['ntrain'].min()
            null_run = p_df[(p_df['ntrain'] == min_n) & (p_df['representation'] == best_model_run['representation']) & (p_df['kernel'] == best_model_run['kernel'])]
            
            if not null_run.empty:
                null_error = null_run.iloc[0]['test_mae']
            else:
                null_error = p_df['test_mae'].max()
                
            if model_error > 0:
                ratio = null_error / model_error
                diff_props.append({'Dataset': best_model_run['dataset'], 'Property': p, 'Ratio (Null/Model)': ratio, 'Test MAE': model_error})
                
        if diff_props:
            diff_df = pd.DataFrame(diff_props)
            diff_df = diff_df.sort_values('Ratio (Null/Model)', ascending=True)
            st.dataframe(diff_df, width='stretch')
        else:
            st.info("Insufficient data to calculate difficult properties.")

    with tab2:
        st.header("Data Exploration with PyGWalker")
        st.write("Filter the dataset before exploring in PyGWalker:")
        
        col1, col2 = st.columns(2)
        with col1:
            explore_dataset = st.selectbox(
                "Filter by Dataset", 
                options=["All"] + sorted(df['dataset'].unique().tolist())
            )
        with col2:
            sample_rate = st.slider("Sample Rate (%)", min_value=1, max_value=100, value=100)
            
        explore_df = df.copy()
        if explore_dataset != "All":
            explore_df = explore_df[explore_df['dataset'] == explore_dataset]
            
        if sample_rate < 100:
            explore_df = explore_df.sample(frac=sample_rate/100.0, random_state=42)
            
        if st.button("Generate PyGWalker Visualization"):
            with st.spinner("Generating PyGWalker HTML..."):
                walker_html = pyg.to_html(explore_df, default_tab="data")
                st.iframe(walker_html, height=800)

# --- Old Graphic Components Execution (Commented Out) ---
# if not all_reps:
#     st.warning("No data found in Archieve directory. Please run `dashboard_manager.py` and then execute `nc-krr` commands to generate results.")
# else:
#     render_leaderboard(experiment_data)
#     render_medaillienspiegel(experiment_data)
#
#     render_pie_matrix(
#         all_reps=all_reps, 
#         all_props=all_props, 
#         datasets=datasets, 
#         completed_combinations=completed_combinations, 
#         json_directory="Archieve"
#     )
#
#     if "selected_plots" not in st.session_state:
#         st.session_state.selected_plots = []
#
#     if st.session_state.get("trigger_dialog", False):
#         st.session_state.trigger_dialog = False
#         render_learning_curves(experiment_data, lit_data)
#
#     render_heatmap(experiment_data)
#     render_heat_properties(experiment_data)
#     render_all_learning_curves_bottom(experiment_data)
