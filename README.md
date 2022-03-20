# Feature-importance-and-selection

## Objective
The goal of this project is to explore the various feature importance algorithms and how to select features in a model.

### Spearman's rank correlation coefficient
The simplest technique to identify important regression features is to rank them by their Spearman's rank correlation coefficient; the feature with the largest coefficient is taken to be the most important. This method is measuring single-feature relevance importance and works well for independent features, but suffers in the presence of codependent features. Groups of features with similar relationships to the response variable receive the same or similar ranks, even though just one should be considered important.

### PCA
Another possibility is to use principle component analysis (PCA), which operates on just the X explanatory matrix. PCA transforms data into a new space characterized by eigenvectors and identifies features that explain the most variance in the new space. If the first principal component covers a large percentage of the variance, the "loads" associated with that component can indicate importance of features in the original X space.

### Model-based importance strategies
- [permutation importance](https://explained.ai/rf-importance/index.html#4)
- [drop column importance](https://explained.ai/rf-importance/index.html#5)

## Installation
<pre>
pip install spicy
pip install -U scikit-learn
pip install shap
pip install pca
pip install xgboost
</pre>
