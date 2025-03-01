import numpy as np
import matplotlib.pyplot as plt


def JS_distance(p, q):
    # Compute the mixture distribution M
    m = 0.5 * (np.array(p) + np.array(q))
    
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return np.sqrt(js_div)



def kl_divergence(p, q):

    p = np.array(p)
    q = np.array(q)

    # Normalize the distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Avoid division by zero by adding a small constant
    q += 1e-10
    p += 1e-10

    return np.sum(p * np.log(p / q))



import numpy as np

def k_center_greedy(plan_cost_list, k, distance_func, first_plan=None, seed=42):
    """
    Greedy K-Center Algorithm using a custom distance function.

    Parameters:
        plan_cost_list: list of distributions (each a vector)
        k: int - number of centers to select
        distance_func: function - a function that computes distance between two plans (JS_distance/KL_divergence)
        seed: int - optional random seed for reproducibility

    Returns:
        centers: list of selected center indices
        assignments: dict where key is the center index and value is a list of indices of assigned plans
    """
    np.random.seed(seed)  # Ensures reproducibility
    n = len(plan_cost_list)
    if k > n:
        centers = list(range(0, len(plan_cost_list)))
        assignments = {center: [center] for center in centers}
        return centers, assignments
    
    # Initialize: Select a random first center
    if first_plan is None:
        centers = [np.random.choice(n, 1)[0]]
    else:
        centers = [first_plan]
    distances = np.full(n, np.inf)  # Initialize distances to infinity

    for _ in range(1, k):
        # Update distances to the nearest selected center
        for i in range(n):
            distances[i] = min(distances[i], distance_func(plan_cost_list[i], plan_cost_list[centers[-1]]))

        # Select the point that is farthest from its closest center
        new_center = np.argmax(distances)
        centers.append(new_center)

    # Initialize assignment dictionary with centers as keys and empty lists for indices
    assignments = {center: [] for center in centers}

    # Assign each plan to the nearest center (store indices instead of plan itself)
    for i in range(n):
        closest_center_idx = np.argmin([distance_func(plan_cost_list[i], plan_cost_list[c]) for c in centers])
        closest_center = centers[closest_center_idx]  # Get the index of the center
        assignments[closest_center].append(i)  # Store index instead of plan itself

    return centers, assignments





def plot_all_cost_distribution(all_cost_list, sort=False, labels=None, anchor=None, file_name="aaa.pdf"):
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_colors = len(default_colors)
    
    def get_color(plan_id):
        try:
            pid_int = int(plan_id)
            color_idx = pid_int % n_colors
        except ValueError:
            color_idx = hash(plan_id) % n_colors
        return default_colors[color_idx]
    
    fig, ax1 = plt.subplots(figsize=(10, 10))
    
    if labels is None:
        for i, cost_list in enumerate(all_cost_list):
            if sort:
                cost_list = sorted(cost_list)
            plan_id = i
            ax1.plot(
                cost_list,
                label=str(plan_id),  
                linestyle='-',
                marker='o',
                markersize=5,
                linewidth=0.5,
                color=get_color(plan_id)
            )
    else:
        anchor_cost_list = None
        for i, cost_list in enumerate(all_cost_list):
            if sort:
                cost_list = sorted(cost_list)
            
            plan_id = labels[i]
            if anchor is not None and plan_id == anchor:
                anchor_cost_list = cost_list
                continue
            else:
                ax1.plot(
                    cost_list,
                    label=str(plan_id),
                    linestyle='--',
                    marker='.',
                    markersize=4,
                    linewidth=0.5,
                    # color=get_color(plan_id)
                    color='lightgrey'
                )
        ax1.plot(
                    anchor_cost_list,
                    label=str(plan_id),
                    linestyle='-',
                    marker='o',
                    markersize=4,
                    linewidth=1,
                    # color='cornflowerblue'
                    color = '#1f77b4'
                )
    ax1.set_ylabel("Log-based Plan Cost", fontsize=30)
    ax1.set_ylim((1000, 1000000000))
    ax1.tick_params(axis='y', labelsize=30)
    ax1.tick_params(axis='x', labelsize=30)

    if sort:
        plt.title("Plan Cost (Sorted) Distribution by samples", fontsize=25)
    else:
        if labels is None:
            plt.title("Plan Cost Distribution", fontsize=25)
        else:
            plt.title("Plan Cost Distribution", fontsize=25)

    for xval in range(len(all_cost_list[0])):
        if xval % 50 == 0:
            ax1.axvline(x=xval, color='grey', linestyle='--', linewidth=1.0, alpha=0.8)

    # Legend (commented out in your original code, but you can enable it if desired)
    handles, label_names = ax1.get_legend_handles_labels()
    unique_handles, unique_labels = [], []
    for h, lab in zip(handles, label_names):
        if lab not in unique_labels:
            unique_labels.append(lab)
            unique_handles.append(h)
    # ax1.legend(unique_handles, unique_labels, loc="upper left", fontsize=15)

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()



def plot_2d_matrix(matrix, id_list=None, filename='plan_cost_KL_matrix.pdf'):
    matrix = np.array(matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')  # You can choose different colormaps
    plt.colorbar(label='Value')
    plt.title('Relative KL of each pair of plan -- Visualization')
    plt.xlabel('Plan ID')
    plt.ylabel('Plan ID')
    
    if id_list is not None:  # Check if id_list is provided
        plt.xticks(ticks=np.arange(matrix.shape[1]), labels=id_list)
        plt.yticks(ticks=np.arange(matrix.shape[0]), labels=id_list)
    else:
        plt.xticks(ticks=np.arange(matrix.shape[1]), labels=np.arange(matrix.shape[1]))
        plt.yticks(ticks=np.arange(matrix.shape[0]), labels=np.arange(matrix.shape[0]))
    
    # Save the figure as a PDF
    plt.savefig(filename, format='pdf')
    plt.close()


def reduce_matrix(matrix, target_rows=5):
    # Convert the 2D list to a NumPy array for easier manipulation
    matrix = np.array(matrix)
    ori_ids = list(range(len(matrix)))
    size = len(matrix)
    for i in range(len(matrix)):
        matrix[i][i] = float('inf')
    while matrix.shape[0] > target_rows:
        # Find the index of the minimum value
        min_index = np.unravel_index(np.argmin(matrix), matrix.shape)
        row_to_remove = min_index[0]
        col_to_remove = min_index[0]
        # print(f"# Reduce plan: we are removing {ori_ids[row_to_remove]}, because of {ori_ids[min_index[0]]} and {ori_ids[min_index[1]]} are similar")
        # Remove the entire row and column
        matrix = np.delete(matrix, row_to_remove, axis=0)  # Remove the row
        matrix = np.delete(matrix, col_to_remove, axis=1)  # Remove the column
        ori_ids = np.delete(ori_ids, row_to_remove)

    return matrix, ori_ids
