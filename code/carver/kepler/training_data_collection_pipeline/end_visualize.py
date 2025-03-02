import os
import matplotlib.pyplot as plt
import csv
from absl import app
from absl import flags

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


def load_results_from_csv(model_result_csv):
    results = []
    with open(model_result_csv, mode='r') as file:
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
    return results


def expand_results_by_frequency(results):
    expanded_results = []
    for result in results:
        expanded_results.extend([result] * result['frequency'])  # Expand based on frequency
    return expanded_results


def sort_results_by_default_latency(results):
    return sorted(results, key=lambda x: x['default_latency'])


def plot_latency_with_confidence(output_dir, query_id, hinted_latency, default_latency, confidence, title, file_suffix):
    x_values = range(len(hinted_latency))  # X-axis values for plotting

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_values, hinted_latency, color='blue', marker='o', label='Robust Plan Latency')
    ax1.plot(x_values, default_latency, color='orange', marker='o', label='Original Latency')

    ax1.set_xlabel('Query Index')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title(title)
    
    # Plot confidence as a red line on the secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x_values, confidence, color='red', marker='o', label='Confidence')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)
    
    # Combine legends from both ax1 and ax2
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.savefig(f'{output_dir}/{query_id}_{file_suffix}.png')
    plt.close(fig)
    print(f"Plot saved: {output_dir}/{query_id}_{file_suffix}.png")


# Plot full (use expanded_result: result * frequency)
def plot_full_with_latency(output_dir, query_id, expanded_results, sorted=False):
    hinted_latency = [result['hinted_latency'] for result in expanded_results]
    default_latency = [result['default_latency'] for result in expanded_results]
    confidence = [result['confidence'] for result in expanded_results]
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
    
    # plot latency with confidence
    confidence_title = f"{query_id} Hinted vs Default Latency with Confidence (Sorted by Default Latency, All Data Expanded)" if sorted else f"{query_id} Hinted vs Default Latency with Confidence (All Data Expanded)"
    file_suffix = 'full_sorted_latency_with_confidence' if sorted else 'full_latency_with_confidence'
    plot_latency_with_confidence(
        output_dir,
        query_id,
        hinted_latency,
        default_latency,
        confidence,
        title=confidence_title,
        file_suffix=file_suffix
    )


# Plot chunk (use original distinct result: result)
def plot_static_chunks_sorted(output_dir, query_id, results, chunk_size=100, split_threshold=0.2):
    sorted_results = sort_results_by_default_latency(results)

    for i in range(0, len(sorted_results), chunk_size):
        chunk = sorted_results[i:i + chunk_size]
        hinted_latency = [result['hinted_latency'] for result in chunk]
        default_latency = [result['default_latency'] for result in chunk]
        confidence = [result['confidence'] for result in chunk]

        # Calculate percentage change in default latency
        max_latency = max(default_latency)
        min_latency = min(default_latency)
        percentage_change = (max_latency - min_latency) / min_latency if min_latency > 0 else 1

        if len(chunk) >= 50 and percentage_change > split_threshold:
            # Split the chunk into two and plot separately if percentage change is large
            mid_point = len(chunk) // 2
            plot_latency_with_confidence(
                output_dir,
                query_id,
                hinted_latency[:mid_point],
                default_latency[:mid_point],
                confidence[:mid_point],
                title=f'{query_id} Hinted vs Default Latency (Sorted, Chunk {i // chunk_size + 1}_Part_1)',
                file_suffix=f'chunk_{i // chunk_size + 1}_part_1_sorted_latency'
            )

            plot_latency_with_confidence(
                output_dir,
                query_id,
                hinted_latency[mid_point:],
                default_latency[mid_point:],
                confidence[mid_point:],
                title=f'{query_id} Hinted vs Default Latency (Sorted, Chunk {i // chunk_size + 1}_Part_2)',
                file_suffix=f'chunk_{i // chunk_size + 1}_part_2_sorted_latency'
            )
        else:
            # Plot as a single chunk if percentage change is small
            plot_latency_with_confidence(
                output_dir,
                query_id,
                hinted_latency,
                default_latency,
                confidence,
                title=f'{query_id} Hinted vs Default Latency (Sorted by Default Latency, Chunk {i // chunk_size + 1})',
                file_suffix=f'chunk_{i // chunk_size + 1}_sorted_latency'
            )



def main(unused_arg):
    output_dir = _OUTPUT_DIR.value
    query_id = _QUERY_ID.value
    model_result_csv = _MODEL_RESULT_CSV_FILE.value
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results from CSV
    results = load_results_from_csv(model_result_csv)
    
    # Expand results based on frequency for the full plot
    expanded_results = expand_results_by_frequency(results)
    
    # Plot the full expanded result
    plot_full_with_latency(output_dir, query_id, expanded_results)
    
    # Plot the full with sorted expanded result
    expanded_results = sort_results_by_default_latency(expanded_results)
    plot_full_with_latency(output_dir, query_id, expanded_results, sorted=True)

    # Plot sorted static 100-row chunks without expanding
    plot_static_chunks_sorted(output_dir, query_id, results, chunk_size=100)


if __name__ == "__main__":
    app.run(main)
