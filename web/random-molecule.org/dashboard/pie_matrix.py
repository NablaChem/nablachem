import streamlit as st
import plotly.graph_objects as go

# Design constants
PALETTE = ["#1F3664", "#D59131", "#931621", "#2C8C99", "#42D9C8"]
DARK_GREY = "#202124"
LIGHT_GREY = "#D3D3D3"
FONT_FAMILY = "'Nunito', sans-serif"

def render_pie_matrix(all_reps, all_props, datasets, completed_combinations, json_directory):
    # Map datasets to specific colors to ensure visual consistency
    dataset_color_map = {ds: PALETTE[i % len(PALETTE)] for i, ds in enumerate(datasets)}
    
    with st.container(border=True):
        st.write("### Experimental Matrix")
        st.write(f"Scanning directory: `{json_directory}`")
        
        for rep in all_reps:
            rep_dimmed = rep in st.session_state.dimmed_reps
            rep_color = LIGHT_GREY if rep_dimmed else DARK_GREY
            rep_icon = ":material/visibility_off:" if rep_dimmed else ":material/visibility:"
            
            row_cols = st.columns([1] + [2] * len(all_props)) 
            
            with row_cols[0]:
                sub_col_text, sub_col_btn = st.columns([3, 2])
                with sub_col_text:
                    st.markdown(f"<br><br><div style='color: {rep_color};'><b>{rep}</b></div>", unsafe_allow_html=True)
                with sub_col_btn:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("", icon=rep_icon, key=f"toggle_rep_{rep}", help=f"Toggle {rep}", type="tertiary"):
                        if rep_dimmed:
                            st.session_state.dimmed_reps.remove(rep)
                        else:
                            st.session_state.dimmed_reps.append(rep)
                        st.rerun()
                
            for i, prop in enumerate(all_props):
                prop_dimmed = prop in st.session_state.dimmed_props
                is_dimmed = rep_dimmed or prop_dimmed
                
                with row_cols[i + 1]:
                    available_datasets = [ds for ds in datasets if (rep, prop, ds) in completed_combinations]
                    
                    if available_datasets:
                        slice_colors = [dataset_color_map[ds] for ds in available_datasets]

                        fig = go.Figure(data=[go.Pie(
                            labels=available_datasets, 
                            values=[1] * len(available_datasets), 
                            textinfo='label',
                            hoverinfo='skip',
                            opacity=0.2 if is_dimmed else 1.0,
                            marker=dict(colors=slice_colors, line=dict(color='#FFFFFF', width=1))
                        )])
                        
                        fig.update_layout(
                            margin=dict(t=10, b=10, l=10, r=10),
                            showlegend=False,
                            height=120,
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)',  
                            font=dict(family=FONT_FAMILY, color=LIGHT_GREY if is_dimmed else DARK_GREY)
                        )
                        
                        event_data = st.plotly_chart(
                            fig, 
                            key=f"{rep}_{prop}", 
                            on_select="rerun", 
                            selection_mode=('points', 'multiple'),
                            width="stretch",
                            config={'displayModeBar': False} 
                        )
                    else:
                        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)

        # Render X-axis columns below the grid
        x_axis_cols = st.columns([1] + [2] * len(all_props))
        for i, prop in enumerate(all_props):
            prop_dimmed = prop in st.session_state.dimmed_props
            prop_color = LIGHT_GREY if prop_dimmed else DARK_GREY
            prop_icon = ":material/visibility_off:" if prop_dimmed else ":material/visibility:"
            
            with x_axis_cols[i + 1]:
                sub_col_text, sub_col_btn = st.columns([3, 2])
                with sub_col_text:
                    st.markdown(f"<div style='text-align: right; padding-top: 10px; color: {prop_color};'><b>{prop}</b></div>", unsafe_allow_html=True)
                with sub_col_btn:
                    if st.button("", icon=prop_icon, key=f"toggle_prop_{prop}", help=f"Toggle {prop}", type="tertiary"):
                        if prop_dimmed:
                            st.session_state.dimmed_props.remove(prop)
                        else:
                            st.session_state.dimmed_props.append(prop)
                        st.rerun()