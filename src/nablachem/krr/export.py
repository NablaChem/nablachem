import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Conversion factor from mHa to kcal/mol
MHA_TO_KCAL = 0.6275095


def load_data(filename):
    """Load data from JSON file"""
    with open(filename, "r") as f:
        data = json.load(f)
    return data.get("learning_curve", [])


def create_learning_curve_plot(learning_curve_data):
    """Create learning curve plot"""
    training_sizes = []
    val_rmse = []
    test_rmse = []
    val_mae = []
    test_mae = []
    nullmodel_rmse = None

    for entry in learning_curve_data:
        ntrain = entry["ntrain"]
        if ntrain == 1:
            nullmodel_rmse = entry["test_rmse"]
        else:
            training_sizes.append(ntrain)
            val_rmse.append(entry["val_rmse"])
            test_rmse.append(entry["test_rmse"])
            val_mae.append(entry["val_mae"])
            test_mae.append(entry["test_mae"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=training_sizes,
            y=val_rmse,
            mode="lines+markers",
            name="val RMSE",
            line=dict(color="#1f77b4", width=2),
            marker=dict(symbol="circle", size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=training_sizes,
            y=test_rmse,
            mode="lines+markers",
            name="test RMSE",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(symbol="square", size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=training_sizes,
            y=val_mae,
            mode="lines",
            name="val MAE",
            line=dict(color="#1f77b4", width=2),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=training_sizes,
            y=test_mae,
            mode="lines",
            name="test MAE",
            line=dict(color="#ff7f0e", width=2),
            showlegend=True,
        )
    )

    if nullmodel_rmse and training_sizes:
        min_training_size = min(training_sizes)
        max_training_size = max(training_sizes)
        fig.add_shape(
            type="line",
            x0=min_training_size * 0.9,
            x1=max_training_size * 1.2,
            y0=nullmodel_rmse,
            y1=nullmodel_rmse,
            line=dict(color="gray", width=1, dash="dash"),
            opacity=0.5,
        )
        fig.add_annotation(
            x=max_training_size * 0.9,
            y=nullmodel_rmse * 1.05,
            text="nullmodel",
            showarrow=False,
            font=dict(color="gray", size=10),
        )

    if training_sizes and val_rmse and test_rmse and val_mae and test_mae:
        all_errors = val_rmse + test_rmse + val_mae + test_mae
        min_error = min(all_errors)
        max_error = max(all_errors)
        max_training_size = max(training_sizes)
        min_training_size = min(training_sizes)

        y_lower_bound = min_error * 0.9
        y_upper_bound = (
            max(nullmodel_rmse, max_error) * 1.05
            if nullmodel_rmse
            else max_error * 1.05
        )

        fig.update_layout(
            xaxis=dict(
                title=dict(text="Training points", font=dict(size=24)),
                type="log",
                range=[
                    np.log10(min_training_size * 0.9),
                    np.log10(max_training_size * 1.2),
                ],
                tickfont=dict(size=20),
                tickmode="array",
                tickvals=training_sizes,
                ticktext=[str(int(n)) for n in training_sizes],
                showline=True,
                linewidth=1,
                linecolor="black",
            ),
            yaxis=dict(
                title=dict(text="RMSE/MAE", font=dict(size=24)),
                type="log",
                showgrid=True,
                dtick=1,
                gridcolor="#555",
                ticks="inside",
                minor={"dtick": 0, "showgrid": True, "ticks": "inside"},
                minorloglabels="none",
                range=[np.log10(y_lower_bound), np.log10(y_upper_bound)],
                tickfont=dict(size=20),
                showline=True,
                linewidth=1,
                linecolor="black",
            ),
            legend=dict(x=0.02, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
            width=800,
            height=500,
            showlegend=True,
        )

    return fig


def create_learning_curve_table(learning_curve_data, energy_unit="mHa"):
    """Create DataFrame for learning curve data"""
    unit_factor = MHA_TO_KCAL if energy_unit == "kcal/mol" else 1.0
    
    lc_df = pd.DataFrame(learning_curve_data)
    
    display_lc_df = lc_df.copy()
    display_lc_df["sigma"] = display_lc_df["hyperparameters"].apply(
        lambda x: (
            "inf"
            if x.get("sigma") == float("inf")
            else f"{x.get('sigma', 0):.3e}"
        )
    )
    display_lc_df["lambda"] = display_lc_df["hyperparameters"].apply(
        lambda x: (
            "-"
            if x.get("sigma") == float("inf")
            else f"{x.get('lambda', 0):.3e}"
        )
    )
    
    display_lc_df["val_rmse"] = display_lc_df["val_rmse"] * unit_factor
    display_lc_df["test_rmse"] = display_lc_df["test_rmse"] * unit_factor
    display_lc_df["val_mae"] = display_lc_df["val_mae"] * unit_factor
    display_lc_df["test_mae"] = display_lc_df["test_mae"] * unit_factor
    
    display_lc_df = display_lc_df[
        [
            "ntrain",
            "val_rmse",
            "test_rmse",
            "val_mae",
            "test_mae",
            "sigma",
            "lambda",
            "combinations_tested",
        ]
    ]
    display_lc_df.columns = [
        "ntrain",
        f"val_rmse ({energy_unit})",
        f"test_rmse ({energy_unit})",
        f"val_mae ({energy_unit})",
        f"test_mae ({energy_unit})",
        "sigma",
        "lambda",
        "combinations_tested",
    ]
    
    return display_lc_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_learning_curve.py <json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    output_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else "."
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    print(f"Loading data from {input_file}...")
    learning_curve_data = load_data(input_file)
    
    print("Creating learning curve plot...")
    fig = create_learning_curve_plot(learning_curve_data)
    plot_filename = os.path.join(output_dir, f"{base_name}_learning_curve.png")
    fig.write_image(plot_filename, width=800, height=500, scale=2)
    print(f"✓ Plot saved to: {plot_filename}")
    
    print("Creating learning curve tables...")
    
    table_mha = create_learning_curve_table(learning_curve_data, energy_unit="mHa")
    csv_filename_mha = os.path.join(output_dir, f"{base_name}_learning_curve_mHa.csv")
    table_mha.to_csv(csv_filename_mha, index=False)
    print(f"✓ Table (mHa) saved to: {csv_filename_mha}")
    
    table_kcal = create_learning_curve_table(learning_curve_data, energy_unit="kcal/mol")
    csv_filename_kcal = os.path.join(output_dir, f"{base_name}_learning_curve_kcal.csv")
    table_kcal.to_csv(csv_filename_kcal, index=False)
    print(f"✓ Table (kcal/mol) saved to: {csv_filename_kcal}")
    
    print("\n✓ Export complete!")


if __name__ == "__main__":
    main()