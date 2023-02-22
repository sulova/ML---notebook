# Permutation importance for feature evaluation 


Check about the method: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance
In this example, we compute the permutation importance on the Wisconsin breast cancer dataset using permutation_importance. 
The RandomForestClassifier can easily get about 97% accuracy on a test dataset. Because this dataset contains multicollinear features, 
the permutation importance will show that none of the features are important. One approach to handling multicollinearity is by performing hierarchical 
clustering on the features’ Spearman rank-order correlations, picking a threshold, and keeping a single feature from each cluster.


Next, we plot the tree based feature importance and the permutation importance. The permutation importance plot shows that permuting a
feature drops the accuracy by at most 0.012, which would suggest that none of the features are important. This is
in contradiction with the high test accuracy computed above: some feature must be important. The permutation importance is 
calculated on the training set to show how much the model relies on each feature during training.

![image](https://user-images.githubusercontent.com/44543964/220553648-53ece6be-cd8c-4d89-95c6-94aa195c1433.png)


When features are collinear, permutating one feature will have little effect on the models performance because it can get the same 
information from a correlated feature. One way to handle multicollinear features is by performing hierarchical clustering on the Spearman rank-order 
correlations, picking a threshold, and keeping a single feature from each cluster. First, we plot a heatmap of the correlated features:
![image](https://user-images.githubusercontent.com/44543964/220553684-4c4380e9-9f3b-4142-b4fd-d713c49b7648.png)

