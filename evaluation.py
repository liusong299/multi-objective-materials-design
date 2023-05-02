import matplotlib.pyplot as plt

def plot_pareto_front(data, optimal_PF_indices, title):
    """
    Plot the optimal PF of the dataset.

    Parameters:
    data (np.ndarray): The entire dataset.
    optimal_PF_indices (np.ndarray): Indices of the Pareto optimal points.
    title (str): The title of the plot.
    """
    plt.scatter(data[:, 0], data[:, 1], label="Data Points")
    plt.scatter(data[optimal_PF_indices, 0], data[optimal_PF_indices, 1], c='r', marker='s', label="Pareto Front")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(title)
    plt.legend()
    plt.show()

def cost_function(data, optimal_PF_indices, sub_PF_indices):
    """
    Calculate the cost function as the average distance between the data points on the optimal front
    and their individual closest neighbors in the sub-PF.
    Parameters:
    data (np.ndarray): The entire dataset.
    optimal_PF_indices (np.ndarray): Indices of the Pareto optimal points.
    sub_PF_indices (np.ndarray): Indices of the sub-Pareto optimal points.

    Returns:
    float: The cost function value.
    """
    cost = 0
    for i in optimal_PF_indices:
        min_distance = np.min(np.linalg.norm(data[i] - data[sub_PF_indices], axis=1))
        cost += min_distance
    cost /= len(optimal_PF_indices)
    return cost

def plot_cost_convergence(cost_list, xlabel, ylabel, title):
    """
    Plot the convergence of the cost function.
    Parameters:
    cost_list (list): List of cost function values at each design cycle.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    title (str): The title of the plot.
    """
    plt.plot(cost_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def compare_strategy_efficiency(design_cycle_lists, strategy_names, title):
    """
    Compare the efficiency of different design strategies.
    Parameters:
    design_cycle_lists (list): List of design cycle lists for each strategy.
    strategy_names (list): List of names for each strategy.
    title (str): The title of the plot.
    """
    for i, strategy in enumerate(design_cycle_lists):
        plt.plot(strategy, label=strategy_names[i])

    plt.xlabel("Prior Training Data Size")
    plt.ylabel("Average Design Cycles to Optimal PF")
    plt.title(title)
    plt.legend()
    plt.show()



