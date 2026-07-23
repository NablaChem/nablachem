import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Design constants
PALETTE = ["#1F3664", "#D59131", "#931621", "#2C8C99", "#42D9C8", "#8A2BE2", "#FF4500", "#2E8B57"]
DARK_GREY = "#202124"
FONT_FAMILY = "'Nunito', sans-serif"

def _apply_layout(fig):
    fig.update_layout(
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Training Set Size", yaxis_title="Test RMSE",
        height=400, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, color=DARK_GREY)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', zeroline=False)

def render_learning_curves(experiment_data, lit_data):
    selected = st.session_state.get('selected_plots', [])
    if not selected:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Select pie chart slices in the Experimental Matrix above to view learning curves. You can Shift-Click to select multiple slices for comparison!")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.write("### Learning Curves")
        
        # Single selection: show the 5 repeats + null + lit
        if len(selected) == 1:
            plot_config = selected[0]
            st.write(f"**{plot_config}**")
            
            parts = [p.strip() for p in plot_config.split("|")]
            if len(parts) != 3: return
            sel_rep, sel_prop, sel_dataset = parts
            
            matched_data = next((d for d in experiment_data if d["rep"] == sel_rep and d["prop"] == sel_prop and d["dataset"] == sel_dataset), None)
            if not matched_data or not matched_data.get("runs"):
                st.info("No learning curve data found.")
                return
                
            curve_fig = go.Figure()
            line_color = PALETTE[0]
            
            null_model_rmse = None
            
            for run_idx, run_lc in enumerate(matched_data["runs"]):
                if not run_lc: continue
                
                nm_step = next((s for s in run_lc if s["ntrain"] == 1), None)
                if nm_step and null_model_rmse is None:
                    null_model_rmse = nm_step.get("test_rmse")
                    
                valid_steps = sorted([s for s in run_lc if s["ntrain"] > 1], key=lambda x: x["ntrain"])
                if not valid_steps: continue
                
                train_sizes = [s["ntrain"] for s in valid_steps]
                test_rmses = [s["test_rmse"] for s in valid_steps]
                
                curve_fig.add_trace(go.Scatter(
                    x=train_sizes, y=test_rmses, mode='lines+markers', name=f'Run {run_idx+1}',
                    line=dict(color=line_color, width=1.5), opacity=0.6, showlegend=False
                ))
            
            if null_model_rmse:
                curve_fig.add_hline(y=null_model_rmse, line_dash="dash", line_color="gray", annotation_text="Null Model", annotation_position="bottom right")
                
            lit_metrics = lit_data.get("datasets", {}).get(sel_dataset, {}).get(sel_prop, {})
            if "RMSE" in lit_metrics:
                curve_fig.add_hline(y=lit_metrics["RMSE"], line_dash="dot", line_color="black", annotation_text=f"{lit_metrics.get('source', 'Literature')} Baseline", annotation_position="top right")
            
            _apply_layout(curve_fig)
            st.plotly_chart(curve_fig, width="stretch", config={'displayModeBar': False})
            
        else:
            # Comparison mode: multiple selections
            st.write("**Comparison Mode**")
            st.write("Showing the *best* performing run for each selected model.")
            
            curve_fig = go.Figure()
            
            for idx, plot_config in enumerate(selected):
                parts = [p.strip() for p in plot_config.split("|")]
                if len(parts) != 3: continue
                sel_rep, sel_prop, sel_dataset = parts
                
                matched_data = next((d for d in experiment_data if d["rep"] == sel_rep and d["prop"] == sel_prop and d["dataset"] == sel_dataset), None)
                if not matched_data or not matched_data.get("runs"): continue
                
                # Find the best run based on lowest final RMSE
                best_run = None
                best_final_rmse = float('inf')
                
                for run_lc in matched_data["runs"]:
                    if not run_lc: continue
                    valid_steps = sorted([s for s in run_lc if s["ntrain"] > 1], key=lambda x: x["ntrain"])
                    if not valid_steps: continue
                    final_rmse = valid_steps[-1].get("test_rmse", float('inf'))
                    if final_rmse < best_final_rmse:
                        best_final_rmse = final_rmse
                        best_run = valid_steps
                
                if best_run:
                    train_sizes = [s["ntrain"] for s in best_run]
                    test_rmses = [s["test_rmse"] for s in best_run]
                    line_color = PALETTE[idx % len(PALETTE)]
                    
                    curve_fig.add_trace(go.Scatter(
                        x=train_sizes, y=test_rmses, mode='lines+markers', name=f"{plot_config}",
                        line=dict(color=line_color, width=2.5)
                    ))
            
            _apply_layout(curve_fig)
            curve_fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(curve_fig, width="stretch", config={'displayModeBar': False})