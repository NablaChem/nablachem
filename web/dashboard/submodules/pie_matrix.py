import streamlit as st
import plotly.graph_objects as go
import re

# Design constants
PALETTE = ["#1F3664", "#D59131", "#931621", "#2C8C99", "#42D9C8"]
DARK_GREY = "#202124"
LIGHT_GREY = "#D3D3D3"
FONT_FAMILY = "'Nunito', sans-serif"

# Fallback for fragment
if hasattr(st, "fragment"):
    fragment_decorator = st.fragment
elif hasattr(st, "experimental_fragment"):
    fragment_decorator = st.experimental_fragment
else:
    def fragment_decorator():
        def decorator(func):
            return func
        return decorator

@fragment_decorator()
def render_pie_matrix(all_reps, all_props, datasets, completed_combinations, json_directory):
    # Map datasets to specific colors to ensure visual consistency
    dataset_color_map = {ds: PALETTE[i % len(PALETTE)] for i, ds in enumerate(datasets)}
    
    with st.container(border=True):
        st.write("### Experimental Matrix")
        st.write(f"Scanning directory: `{json_directory}`")
        
        for rep in all_reps:
            row_cols = st.columns([1] + [2] * len(all_props)) 
            
            with row_cols[0]:
                st.markdown(f"<br><br><div style='color: {DARK_GREY};'><b>{rep}</b></div>", unsafe_allow_html=True)
                
            for i, prop in enumerate(all_props):
                with row_cols[i + 1]:
                    available_datasets = [ds for ds in datasets if (rep, prop, ds) in completed_combinations]
                    
                    if available_datasets:
                        raw_btn_key = f"btn_{rep}_{prop}"
                        btn_key = re.sub(r'[^a-zA-Z0-9_]', '_', raw_btn_key)
                        plot_ids = [f"{rep} | {prop} | {ds}" for ds in available_datasets]
                        
                        if 'selected_plots' not in st.session_state:
                            st.session_state.selected_plots = []
                            
                        selected = all(pid in st.session_state.selected_plots for pid in plot_ids)
                        
                        # Generate CSS pie chart gradient
                        n = len(available_datasets)
                        css_gradient = []
                        for idx, ds in enumerate(available_datasets):
                            color = dataset_color_map[ds]
                            start_pct = (idx / n) * 100
                            end_pct = ((idx + 1) / n) * 100
                            css_gradient.append(f"{color} {start_pct}% {end_pct}%")
                            
                        bg_css = f"conic-gradient({', '.join(css_gradient)})" if n > 1 else dataset_color_map[available_datasets[0]]
                        opacity = "1.0"
                        border = "3px solid #000000" if selected else "1px solid #FFFFFF"
                        
                        st.markdown(f"""
                        <style>
                        .element-container:has(#span_{btn_key}) + .element-container, 
                        .element-container:has(#span_{btn_key}) + .element-container > div {{
                            display: flex !important;
                            justify-content: center !important;
                            width: 100% !important;
                        }}
                        .element-container:has(#span_{btn_key}) + .element-container button {{
                            background: {bg_css} !important;
                            border-radius: 50% !important;
                            width: 80px !important;
                            height: 80px !important;
                            min-height: 80px !important;
                            opacity: {opacity} !important;
                            border: {border} !important;
                            color: transparent !important;
                            padding: 0 !important;
                            margin: 20px auto !important;
                            display: block !important;
                        }}
                        .element-container:has(#span_{btn_key}) + .element-container button p {{
                            display: none !important;
                        }}
                        </style>
                        <span id="span_{btn_key}"></span>
                        """, unsafe_allow_html=True)
                        
                        if st.button(" ", key=btn_key, help="Datasets: " + ", ".join(available_datasets)):
                            # If it was already the only one selected, maybe unselect it? Or just re-select it
                            # If they click a piece, make it the EXCLUSIVE selection.
                            if selected and len(st.session_state.selected_plots) == len(plot_ids) and all(p in st.session_state.selected_plots for p in plot_ids):
                                st.session_state.selected_plots = []
                            else:
                                st.session_state.selected_plots = list(plot_ids)
                                st.session_state.trigger_dialog = True
                            st.rerun()
                    else:
                        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)

        # Render X-axis columns below the grid
        x_axis_cols = st.columns([1] + [2] * len(all_props))
        for i, prop in enumerate(all_props):
            with x_axis_cols[i + 1]:
                # Use absolute positioning to perfectly center the text without stretching the Streamlit column
                label_html = f"""
                <div style="position: relative; width: 100%; height: 40px; padding-top: 10px;">
                    <div style="position: absolute; left: 50%; transform: translateX(-50%); white-space: nowrap; color: {DARK_GREY};">
                        <b>{prop}</b>
                    </div>
                </div>
                """
                st.markdown(label_html, unsafe_allow_html=True)