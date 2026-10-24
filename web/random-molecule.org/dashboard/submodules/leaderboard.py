import streamlit as st

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
def render_leaderboard(experiment_data):
    with st.container(border=True):
        st.write("### Leaderboard")
        
        # Extract unique, sorted options for the dropdowns
        datasets = sorted(list(set(d.get("dataset") for d in experiment_data if d.get("dataset"))))
        properties = sorted(list(set(d.get("prop") for d in experiment_data if d.get("prop"))))
        kernels = sorted(list(set(d.get("kernel") for d in experiment_data if d.get("kernel"))))
        
        all_train_points = set()
        for d in experiment_data:
            for run in d.get("runs", []):
                for step in run:
                    if step.get("ntrain", 0) > 1:
                        all_train_points.add(step["ntrain"])
        train_points = sorted(list(all_train_points))
        
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
        filtered_data = []
        err_key = "test_mae" if selected_error == "MAE" else "test_rmse"
        
        for d in experiment_data:
            if d.get("dataset") == selected_dataset and d.get("prop") == selected_prop and d.get("kernel") == selected_kernel:
                if selected_train == "All":
                    if d.get(selected_error) is not None:
                        filtered_data.append(d)
                else:
                    best_error = float('inf')
                    for run in d.get("runs", []):
                        for step in run:
                            if step.get("ntrain") == selected_train:
                                val = step.get(err_key)
                                if val is not None and val < best_error:
                                    best_error = val
                    if best_error != float('inf'):
                        d_copy = dict(d)
                        d_copy[selected_error] = best_error
                        filtered_data.append(d_copy)

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