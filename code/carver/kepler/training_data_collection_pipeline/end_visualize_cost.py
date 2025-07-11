import os
import matplotlib.pyplot as plt
import csv
from absl import app
from absl import flags
import pandas as pd

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")
_MODEL_RESULT_CSV_FILE = flags.DEFINE_string(
    "model_result_csv_file", None,
    "CSV file regarding kepler model result.")
flags.mark_flag_as_required("model_result_csv_file")
_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")
_COST_FILE = flags.DEFINE_string(
    "cost_file", None,
    "CSV file regarding cost.")
flags.mark_flag_as_required("cost_file")


def load_results_from_csv(model_result_csv, cost_file_csv):
    # Load cost_file and model_result as DataFrames
    cost_df = pd.read_csv(cost_file_csv)
    model_result_df = pd.read_csv(model_result_csv)

    # Ensure columns like params, plan_content, etc., are strings for matching
    cost_df['key'] = cost_df.apply(
        lambda row: f"{row['params']}_{row['frequency']}_{row['run']}_{row['confidence']}_{row['plan_id']}_{row['plan_content']}", axis=1
    )
    model_result_df['key'] = model_result_df.apply(
        lambda row: f"{row['params']}_{row['frequency']}_{row['run']}_{row['confidence']}_{row['plan_id']}_{row['plan_content']}", axis=1
    )
    
    # Merge model results with cost data on the 'key'
    merged_df = pd.merge(model_result_df, cost_df, on='key', suffixes=('_model', '_cost'))
    
    # Combine results into a list of dictionaries
    combined_results = merged_df[[
        'params_model', 'frequency_model', 'run_model', 'hinted_latency', 'default_latency',
        'confidence_model', 'plan_id_model', 'plan_content_model', 'hinted_cost', 'default_cost'
    ]].rename(columns={
        'params_model': 'params',
        'frequency_model': 'frequency',
        'run_model': 'run',
        'confidence_model': 'confidence',
        'plan_id_model': 'plan_id',
        'plan_content_model': 'plan_content'
    }) # type: ignore
    
    # Convert the DataFrame to a list of dictionaries
    results = combined_results.to_dict(orient='records')
    
    return results


def expand_results_by_frequency(results):
    expanded_results = []
    for result in results:
        expanded_results.extend([result] * result['frequency'])  # Expand based on frequency
    return expanded_results


def sort_results_by_default_latency(results):
    return sorted(results, key=lambda x: x['default_latency'])


# Plot full (use expanded_result: result * frequency)
def plot_full_with_latency(output_dir, query_id, expanded_results, sorted=False):
    # Extract data
    hinted_latency = [result['hinted_latency'] for result in expanded_results]
    default_latency = [result['default_latency'] for result in expanded_results]
    hinted_cost = [result['hinted_cost'] for result in expanded_results]
    default_cost = [result['default_cost'] for result in expanded_results]
    x_values = range(len(hinted_latency))  # X-axis values for plotting

    if sorted:
        # Combined plot for latency and cost when sorted=True
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot latency on the first y-axis
        ax1.plot(x_values, hinted_latency, color='blue', marker='o', label='Robust Plan Latency')
        ax1.plot(x_values, default_latency, color='orange', marker='o', label='Original Latency')
        ax1.set_xlabel('Query Index')
        ax1.set_ylabel('Latency (ms)')
        title = f"{query_id} Hinted vs Default Latency (Sorted by Default Latency, All Data Expanded)"
        ax1.set_title(title)

        # Add a second y-axis for cost
        ax2 = ax1.twinx()
        ax2.plot(x_values, hinted_cost, color='darkblue', marker='x', label='Robust Plan Cost')
        ax2.plot(x_values, default_cost, color='red', marker='x', label='Original Cost')
        ax2.set_ylabel('Cost')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        # Save combined figure
        figure_name = f'{output_dir}/{query_id}_full_sorted_latency_cost.png'
        plt.savefig(figure_name)
        plt.close(fig)
        print(f"Plot saved: {figure_name}")

    else:
        # Generate latency-only plot when sorted=False
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(x_values, hinted_latency, color='blue', marker='o', label='Robust Plan Latency')
        ax1.plot(x_values, default_latency, color='orange', marker='o', label='Original Latency')
        ax1.set_xlabel('Query Index')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title(f"{query_id} Hinted vs Default Latency (All Data Expanded)")
        ax1.legend(loc='upper left')

        # Save latency-only figure
        latency_figure_name = f'{output_dir}/{query_id}_full_latency.png'
        plt.savefig(latency_figure_name)
        plt.close(fig)
        print(f"Latency plot saved: {latency_figure_name}")

        # Generate cost-only plot when sorted=False
        fig, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(x_values, hinted_cost, color='darkblue', marker='x', linestyle='--', label='Robust Plan Cost')
        ax2.plot(x_values, default_cost, color='red', marker='x', linestyle='--', label='Original Cost')
        ax2.set_xlabel('Query Index')
        ax2.set_ylabel('Cost')
        ax2.set_title(f"{query_id} Hinted vs Default Cost (All Data Expanded)")
        ax2.legend(loc='upper left')

        # Save cost-only figure
        cost_figure_name = f'{output_dir}/{query_id}_full_cost.png'
        plt.savefig(cost_figure_name)
        plt.close(fig)
        print(f"Cost plot saved: {cost_figure_name}")

def main(unused_arg):
    output_dir = _OUTPUT_DIR.value
    query_id = _QUERY_ID.value
    model_result_csv = _MODEL_RESULT_CSV_FILE.value
    cost_file_csv = _COST_FILE.value
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results from CSV
    results = load_results_from_csv(model_result_csv, cost_file_csv)
    
    # Expand results based on frequency for the full plot
    expanded_results = expand_results_by_frequency(results)
    
    # Plot the full expanded result
    plot_full_with_latency(output_dir, query_id, expanded_results)
    
    # Plot the full with sorted expanded result
    expanded_results = sort_results_by_default_latency(expanded_results)
    plot_full_with_latency(output_dir, query_id, expanded_results, sorted=True)


if __name__ == "__main__":
    app.run(main)
