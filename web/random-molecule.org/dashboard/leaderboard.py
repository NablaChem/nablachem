import streamlit as st

def render_leaderboard(experiment_data):
    with st.container(border=True):
        st.write("### Leaderboard")
        
        # Extract unique, sorted options for the dropdowns
        datasets = sorted(list(set(d.get("dataset") for d in experiment_data if d.get("dataset"))))
        properties = sorted(list(set(d.get("prop") for d in experiment_data if d.get("prop"))))
        kernels = sorted(list(set(d.get("kernel") for d in experiment_data if d.get("kernel"))))
        train_points = sorted(list(set(d.get("train_size") for d in experiment_data if d.get("train_size") is not None)))
        
        errors = ["MAE", "RMSE"]

        # Provide fallback options if the directory is empty
        if not datasets: datasets = ["Unknown"]
        if not properties: properties = ["Unknown"]
        if not kernels: kernels = ["Unknown"]
        if not train_points: train_points = ["All"]

        # Top controls using 5 equal columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            selected_dataset = st.selectbox("Dataset", datasets, key="lb_dataset")
        with col2:
            selected_prop = st.selectbox("Property", properties, key="lb_prop")
        with col3:
            selected_kernel = st.selectbox("Kernel", kernels, key="lb_kernel")
        with col4:
            selected_train = st.selectbox("Training Points", train_points, key="lb_train")
        with col5:
            selected_error = st.selectbox("Error Metric", errors, key="lb_error")

        # Filter data
        filtered_data = [
            d for d in experiment_data
            if d.get("dataset") == selected_dataset
            and d.get("prop") == selected_prop
            and d.get("kernel") == selected_kernel
            and (selected_train == "All" or d.get("train_size") == selected_train)
            and d.get(selected_error) is not None
        ]

        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        if not filtered_data:
            st.markdown("No data available for this combination.")
        else:
            sorted_data = sorted(filtered_data, key=lambda x: x[selected_error])
            top_3 = sorted_data[:3]

            # Render top 3 using the matching 5-column structure
            for i, entry in enumerate(top_3, 1):
                rep_name = entry.get("rep", "Unknown")
                error_val = entry.get(selected_error, 0.0)
                
                # Match the 5-column layout above (col1=Medal, col2-col4=Representation, col5=Error Metric)
                r_col1, r_col2, r_col5 = st.columns([1, 3, 1])
                
                with r_col1:
                    emoji_font_stack = "font-family: 'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', sans-serif; font-size: 1.4rem;"
                    if i == 1:
                        st.markdown(f'<span style="{emoji_font_stack}">🥇</span>', unsafe_allow_html=True)
                    elif i == 2:
                        st.markdown(f'<span style="{emoji_font_stack}">🥈</span>', unsafe_allow_html=True)
                    elif i == 3:
                        st.markdown(f'<span style="{emoji_font_stack}">🥉</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{i}**")
                        
                with r_col2:
                    st.markdown(f"{rep_name}")
                    
                with r_col5:
                    # Aligns directly under col5 (the Error Metric selectbox)
                    st.markdown(f"**{error_val:.6f}**")