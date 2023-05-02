import numpy as np

def pareto_front(data):
    """
    Calculate the Pareto front of the dataset.

    Parameters:
    data (np.ndarray): An array containing the objectives.

    Returns:
    np.ndarray: Indices of the Pareto optimal points.
    """
    is_dominated = np.any(data[:, None] < data, axis=-1)
    pareto_front_indices = np.where(~is_dominated.any(axis=1))[0]
    return pareto_front_indices

def update_training_data(training_data, new_measurement):
    """
    Update the training data with new measurements.

    Parameters:
    training_data (np.ndarray): The existing training data.
    new_measurement (np.ndarray): The new measurement to be added.

    Returns:
    np.ndarray: The updated training data.
    """
    updated_training_data = np.vstack([training_data, new_measurement])
    return updated_training_data

def average_design_cycles_to_optimal_PF(data, prior_training_data_size, num_repeats, design_strategy_func, model):
    """
    Calculate the average number of design cycles needed to find the optimal PF.

    Parameters:
    data (np.ndarray): The entire dataset.
    prior_training_data_size (int): Size of prior training data.
    num_repeats (int): Number of times to repeat the design process.
    design_strategy_func (callable): The design strategy function.
    model (sklearn estimator): The trained regression model.

    Returns:
    float: The average number of design cycles needed to find the optimal PF.
    """
    design_cycles_list = []
    for _ in range(num_repeats):
        # Randomly select prior training data
        prior_indices = np.random.choice(data.shape[0], prior_training_data_size, replace=False)
        prior_training_data = data[prior_indices]
        remaining_data = np.delete(data, prior_indices, axis=0)

        # Calculate the optimal PF
        optimal_PF_indices = pareto_front(data)

        # Initialize sub-PF
        sub_PF_indices = []

        design_cycles = 0
        while not np.array_equal(optimal_PF_indices, sub_PF_indices):
            selected_material_index = design_strategy_func(remaining_data, model)
            new_measurement = remaining_data[selected_material_index]

            # Update training data and search space
            prior_training_data = update_training_data(prior_training_data, new_measurement)
            remaining_data = np.delete(remaining_data, selected_material_index, axis=0)

            # Update sub-PF
            sub_PF_indices = pareto_front(prior_training_data)

            design_cycles += 1

        design_cycles_list.append(design_cycles)

    avg_design_cycles = np.mean(design_cycles_list)
    return avg_design_cycles
