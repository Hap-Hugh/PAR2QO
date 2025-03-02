import os
import matplotlib.pyplot as plt
import csv
from math import ceil, sqrt
from absl import app
from absl import flags
import numpy as np

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
_ROBUSTNESS = flags.DEFINE_string("robustness", "robustness", "robustness pipeline type")


def load_results_from_csv_files(base_csv_file, robustness):
    results_by_version = {}
    # TODO: check range
    ranging = [1, 2, 3, 4, 6, 7] if robustness == 'category' else range(9)
    for i in ranging:
        csv_file = f"{base_csv_file}_{i}.csv"
        if os.path.exists(csv_file):
            results = []
            with open(csv_file, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    results.append({
                        'params': eval(row['params']),
                        'frequency': int(row['frequency']),
                        'run': row['run'],
                        'hinted_latency': float(row['hinted_latency']),
                        'default_latency': float(row['default_latency']),
                        'hinted_cost': float(row['hinted_cost']),
                        'default_cost': float(row['default_cost']),
                        'confidence': float(row['confidence']),
                        'plan_id': int(row['plan_id']),
                        'plan_content': row['plan_content']
                    })
            results_by_version[f"db_instance_{i}"] = results
    return results_by_version


def plot_subplots(output_dir, query_id, results_by_version):
    num_versions = len(results_by_version)
    rows = cols = ceil(sqrt(num_versions))

    # Plot 1: Confidence and Latency
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for idx, (version, results) in enumerate(results_by_version.items()):
        ax = axes[idx]
        ax2 = ax.twinx()  # Secondary y-axis for confidence

        hinted_latency = [result['hinted_latency'] for result in results]
        default_latency = [result['default_latency'] for result in results]
        confidence = [result['confidence'] for result in results]
        x_values = range(len(hinted_latency))

        # Latency and confidence
        ax.plot(x_values, hinted_latency, marker='o', label='Robust Plan Latency', color='blue')
        ax.plot(x_values, default_latency, marker='o', label='Original Latency', color='orange')
        ax2.plot(x_values, confidence, marker='o', label='Confidence', color='red')

        ax.set_title(f'{version}')
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Latency (ms)')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'{query_id} Latency and Confidence Comparison (Subplots)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/{query_id}_subplots_latency_confidence.png')
    plt.close(fig)
    print(f"Subplots (Latency and Confidence) saved: {output_dir}/{query_id}_subplots_latency_confidence.png")

    # Plot 2: Cost
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for idx, (version, results) in enumerate(results_by_version.items()):
        ax = axes[idx]

        hinted_cost = [result['hinted_cost'] for result in results]
        default_cost = [result['default_cost'] for result in results]
        x_values = range(len(hinted_cost))

        # Cost
        ax.plot(x_values, hinted_cost, marker='x', label='Hinted Cost', color='darkblue')
        ax.plot(x_values, default_cost, marker='x', label='Default Cost', color='red')

        ax.set_title(f'{version}')
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Cost')

        ax.legend(loc='upper left')

    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'{query_id} Cost Comparison (Subplots)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/{query_id}_subplots_cost.png')
    plt.close(fig)
    print(f"Subplots (Cost) saved: {output_dir}/{query_id}_subplots_cost.png")

    # Plot 3: Sorted Default Latency with Latency and Cost
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for idx, (version, results) in enumerate(results_by_version.items()):
        ax = axes[idx]

        # Sort results by default_latency
        sorted_results = sorted(results, key=lambda r: r['default_latency'])
        hinted_latency = [result['hinted_latency'] for result in sorted_results]
        default_latency = [result['default_latency'] for result in sorted_results]
        hinted_cost = [result['hinted_cost'] for result in sorted_results]
        default_cost = [result['default_cost'] for result in sorted_results]
        x_values = range(len(hinted_latency))

        # Latency
        ax.plot(x_values, hinted_latency, marker='o', label='Hinted Latency (Sorted)', color='blue')
        ax.plot(x_values, default_latency, marker='o', label='Default Latency (Sorted)', color='orange')
        # Cost
        ax.plot(x_values, hinted_cost, marker='x', label='Hinted Cost (Sorted)', color='darkblue')
        ax.plot(x_values, default_cost, marker='x', label='Default Cost (Sorted)', color='red')

        ax.set_title(f'{version}')
        ax.set_xlabel('Query Index (Sorted by Default Latency)')
        ax.set_ylabel('Latency and Cost')

        ax.legend(loc='upper left')

    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'{query_id} Sorted Default Latency with Cost (Subplots)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/{query_id}_subplots_sorted_latency_cost.png')
    plt.close(fig)
    print(f"Subplots (Sorted Latency and Cost) saved: {output_dir}/{query_id}_subplots_sorted_latency_cost.png")


def calculate_averages(results_by_version, keys):
    max_queries = max(len(results) for results in results_by_version.values())
    averages = {key: [] for key in keys}

    for query_idx in range(max_queries):
        for key in keys:
            values = [results[query_idx][key] for results in results_by_version.values() if query_idx < len(results)]
            averages[key].append(np.mean(values) if values else 0)

    return [averages[key] for key in keys]


def plot_combined(output_dir, query_id, results_by_version):
    # Plot latency and confidence
    fig, ax = plt.subplots(figsize=(12, 8))
    ax2 = ax.twinx()  # Secondary y-axis for confidence

    avg_hinted_latency, avg_default_latency, avg_confidence = calculate_averages(results_by_version, ['hinted_latency', 'default_latency', 'confidence'])

    x_values = range(len(avg_hinted_latency))
    ax.plot(x_values, avg_hinted_latency, marker='o', label='Average Robust Plan Latency', color='blue')
    ax.plot(x_values, avg_default_latency, marker='o', label='Average Original Latency', color='orange')
    ax2.plot(x_values, avg_confidence, marker='o', label='Average Confidence', color='red')

    ax.set_ylabel('Latency (ms)')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)

    ax.set_xlabel('Query Index')
    ax.set_title(f'{query_id} Latency and Confidence (Combined)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(f'{output_dir}/{query_id}_combined_latency_confidence.png')
    plt.close(fig)
    print(f"Combined (Latency and Confidence) plot saved: {output_dir}/{query_id}_combined_latency_confidence.png")

    # Plot cost
    fig, ax = plt.subplots(figsize=(12, 8))

    avg_hinted_cost, avg_default_cost = calculate_averages(results_by_version, ['hinted_cost', 'default_cost'])

    x_values = range(len(avg_hinted_cost))
    ax.plot(x_values, avg_hinted_cost, marker='x', label='Average Robust Plan Cost', color='darkblue')
    ax.plot(x_values, avg_default_cost, marker='x', label='Average Original Cost', color='red')

    ax.set_ylabel('Cost')
    ax.set_xlabel('Query Index')
    ax.set_title(f'{query_id} Cost (Combined)')
    ax.legend(loc='upper left')

    plt.savefig(f'{output_dir}/{query_id}_combined_cost.png')
    plt.close(fig)
    print(f"Combined (Cost) plot saved: {output_dir}/{query_id}_combined_cost.png")

    # Plot sorted default latency with hinted and default latencies and costs
    sorted_indices = np.argsort(avg_default_latency)
    sorted_hinted_latency = [avg_hinted_latency[i] for i in sorted_indices]
    sorted_default_latency = [avg_default_latency[i] for i in sorted_indices]
    sorted_hinted_cost = [avg_hinted_cost[i] for i in sorted_indices]
    sorted_default_cost = [avg_default_cost[i] for i in sorted_indices]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()  # Secondary y-axis for costs

    x_values = range(len(sorted_indices))

    # Plot latencies
    ax1.plot(x_values, sorted_hinted_latency, marker='o', label='Hinted Latency (Sorted)', color='blue')
    ax1.plot(x_values, sorted_default_latency, marker='o', label='Default Latency (Sorted)', color='orange')

    # Plot costs
    ax2.plot(x_values, sorted_hinted_cost, marker='x', label='Hinted Cost (Sorted)', color='darkblue', linestyle='--')
    ax2.plot(x_values, sorted_default_cost, marker='x', label='Default Cost (Sorted)', color='red', linestyle='--')

    # Set axis labels and limits
    ax1.set_ylabel('Latency (ms)')
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('Query Index (Sorted by Default Latency)')
    ax1.set_title(f'{query_id} Sorted Default Latency: Latency and Cost')

    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Save the figure
    plt.savefig(f'{output_dir}/{query_id}_combined_sorted_latency_cost.png')
    plt.close(fig)
    print(f"Combined (Sorted Latency) plot saved: {output_dir}/{query_id}_combined_sorted_latency_cost.png")



def plot_multiple_versions(output_dir, query_id, results_by_version):
    plot_subplots(output_dir, query_id, results_by_version)
    plot_combined(output_dir, query_id, results_by_version)


def main(unused_arg):
    output_dir = _OUTPUT_DIR.value
    query_id = _QUERY_ID.value
    base_csv_file = _MODEL_RESULT_CSV_FILE.value
    robustness = _ROBUSTNESS.value

    os.makedirs(output_dir, exist_ok=True)

    results_by_version = load_results_from_csv_files(base_csv_file, robustness)

    plot_multiple_versions(output_dir, query_id, results_by_version)

if __name__ == "__main__":
    app.run(main)
    