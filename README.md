# Multi-Objective Materials Design Strategies Project

This project aims to implement and compare various multi-objective optimization design strategies for materials discovery. These design strategies include Maximin, Centroid, Random, Pure Exploitation, and Pure Exploration. We use three materials datasets with different characteristics: Binh-Korn function test dataset, Shape Memory Alloy (SMA) dataset, and two Density Functional Theory (DFT) datasets (MAX phase and Piezoelectric).

The main purpose is to find the optimal Pareto Fronts (PFs) in as few design cycles as possible, starting with a smaller subset of data assumed to be initially known.

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation

Clone this repository and install the required packages using pip:

```bash
git clone https://github.com/liusong299/multi-objective-materials-design.git
cd multi-objective-materials-design
pip install -r requirements.txt
```

## Usage

Run the main script to execute the design process for each dataset and strategy:

```bash
python main.py
```

## Project Structure

The project is divided into several modules:

1. `data_preprocessing.py`: Preprocess the datasets, normalize feature values, and divide data into training and search spaces.
2. `surrogate_model.py`: Build the machine learning regression model (e.g., Gaussian Process, Random Forest) that serves as a surrogate for the true materials property relationships and evaluate its performance.
3. `design_strategies.py`: Implement the various design strategies (Maximin, Centroid, Random, Pure Exploitation, and Pure Exploration).
4. `pareto_front.py`: Determine the optimal Pareto front and sub-Pareto front after each design cycle, update the training data with new measurements, and calculate the average number of design cycles needed to find the optimal PF.
5. `evaluation.py`: Evaluate the performance of each design strategy and visualize the results.
6. `main.py`: Integrate all the modules and execute the entire project workflow.

## Results

The project generates plots of the optimal PF and sub-PF for each dataset, calculates the cost function and plots its convergence, and compares the efficiency of different design strategies.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- Song Liu (liusong299@email.com)

## Acknowledgments

This project is based on the research paper: Gopakumar, A. M., Balachandran, P. V., Xue, D., Gubernatis, J. E., & Lookman, T. (2018). Multi-objective Optimization for Materials Discovery via Adaptive Design. Scientific Reports, 8(1), 1-12. https://doi.org/10.1038/s41598-018-21936-3
