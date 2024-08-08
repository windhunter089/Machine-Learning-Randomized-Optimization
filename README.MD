# Project 2 - Randomized Optimization

## Introduction

This project explores the effectiveness of three randomized optimization algorithms—Randomized Hill Climbing (RHC), Simulated Annealing (SA), and Genetic Algorithms (GA)—in solving optimization problems. The experiments are divided into two parts:
1. Solving the Four Peaks and K-Coloring problems using the `mlrose_hiive` library.
2. Using the optimization algorithms to update the weights of a neural network and comparing the results with traditional backpropagation. The dataset used for this experiment is the Wine Quality dataset.

## Project Structure

- **4Peak.ipynb** and **Kcolor.ipynb**: Notebooks for Experiment 1.
- All other files with `NN` in the name are notebooks for Experiment 2.

## Requirements

The project is done in Python 3.8 with the following libraries:
- `numpy`
- `pandas`
- `torch`
- `skorch`
- `matplotlib`
- `mlrose_hiive`
- `networkx`
- `IPython.core.display`
- `pyperch`

**Note:** `pyperch` cannot be installed via `pip install`. It needs to be manually copied or cloned from [pyperch GitHub repository](https://github.com/jlm429/pyperch).

The dataset for Experiment 2, `Winequality-white.csv`, is used in the analysis.

## Data and Methodology

### Experiment 1: Optimization Problems

#### Four Peaks Problem
The Four Peaks problem involves finding a binary string of length 50 that maximizes an objective function. The goal is to maximize the fitness score of the string solution.

#### K-Coloring Problem
The K-Coloring problem involves assigning colors to the vertices of a graph so that no adjacent nodes have the same color. The goal is to minimize the number of conflicts.

### Experiment 2: Neural Network Weight Optimization

In this experiment, we use the Wine Quality dataset to optimize the weights of a neural network using RHC, SA, and GA, and compare the results with traditional backpropagation.

#### Wine Quality Dataset
- The dataset has 11 features and 4,898 records.
- The target variable is categorized into three classes: low, medium, and high quality.
- The data is split into 80% for training and 20% for testing.
- Five-fold cross-validation is used to assess the performance of the models.

## Models and Results

### Randomized Hill Climbing (RHC)
- **Experiment 1**: RHC performed better in the K-Coloring problem compared to the Four Peaks problem due to the higher chance of finding optimal solutions.
- **Experiment 2**: RHC achieved an accuracy score of 57% but struggled with predicting high-quality wine.

### Simulated Annealing (SA)
- **Experiment 1**: SA showed good performance with fewer function evaluations and less training time compared to RHC and GA.
- **Experiment 2**: SA achieved a lower accuracy score due to limited iterations but demonstrated good potential with more training.

### Genetic Algorithms (GA)
- **Experiment 1**: GA consistently outperformed other methods in both problems, showing low bias and variance.
- **Experiment 2**: GA achieved the highest accuracy score among the optimization algorithms but required significant computational resources.

### Backpropagation
- **Experiment 2**: Backpropagation remained the preferred method for updating neural network weights, achieving an accuracy score of 62%.

## Conclusion

The experiments demonstrated that GA is a robust optimization solution due to its ability to effectively explore the search space and avoid local optima. However, SA proved to be more efficient in terms of computational cost. RHC, while effective in simpler problems, required substantial computational power for more complex problems.

For neural network weight optimization, traditional backpropagation remains the preferred method due to its efficiency and performance, although GA showed promising results with significant computational resources.

## Limitations

- The models did not perform well with imbalanced datasets. Applying SMOTE or other oversampling techniques could improve performance.
- The SA model was underfit and could benefit from more extensive hyperparameter tuning and more iterations.
- The GA model could be trained faster with parallel processing.

## Resources

- [Scikit-learn API Reference](https://scikit-learn.org)
- [Wine Quality Dataset on UCI](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [pyperch GitHub Repository](https://github.com/jlm429/pyperch)
- [mlrose-hiive GitHub Repository](https://github.com/hiive/mlrose)
- [Skorch Documentation](https://skorch.readthedocs.io/en/stable/)

## Author
Feedback and questions, please send to Ted Pham at trungpham89@gmail.com