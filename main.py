import data_preprocessing
import surrogate_model
import design_strategies
import pareto_front
import evaluation

def main():
    # Load and preprocess datasets
    binh_korn_data, sma_data, max_phase_data, piezoelectric_data = data_preprocessing.load_datasets()

    # Train regression models for each dataset
    bk_model = surrogate_model.train_model(binh_korn_data)
    sma_model = surrogate_model.train_model(sma_data)
    max_phase_model = surrogate_model.train_model(max_phase_data)
    piezo_model = surrogate_model.train_model(piezoelectric_data)

    # Define design strategies
    strategies = [design_strategies.maximin_based,
                  design_strategies.centroid_based,
                  design_strategies.random_selection,
                  design_strategies.pure_exploitation,
                  design_strategies.pure_exploration]

    strategy_names = ["Maximin-based", "Centroid-based", "Random", "Pure Exploitation", "Pure Exploration"]

    # Execute the design process for each dataset and strategy
    for data, model in zip([binh_korn_data, sma_data, max_phase_data, piezoelectric_data],
                           [bk_model, sma_model, max_phase_model, piezo_model]):
        optimal_PF_indices = pareto_front.pareto_front(data)
        evaluation.plot_pareto_front(data, optimal_PF_indices, "Dataset Pareto Front")

        design_cycle_lists = []
        for strategy in strategies:
            design_cycles = pareto_front.average_design_cycles_to_optimal_PF(data, 5, 10, strategy, model)
            design_cycle_lists.append(design_cycles)

        evaluation.compare_strategy_efficiency(design_cycle_lists, strategy_names, "Design Strategy Efficiency")

if __name__ == "__main__":
    main()
