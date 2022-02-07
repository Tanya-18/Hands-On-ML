### 1) What is the fundamental idea behind Support Vector Machines?
A Support Vector Machine analyzes data points and constructs a hyperplane that efficiently divides classes. The best hyperplane is the one whose distance to the nearest element of each class is the largest.

### 2) What is a support vector?
The instances on the edge of the decision boundary are known as support vectors. The decision boundary is totally controlled by the support vectors, therefore adding more instances off the margin will have no effect.

### 3) Why is it important to scale the inputs when using SVMs?
 Support Vector Machines are sensitive to feature scales. When using unscaled inputs, features with a larger range of values have a greater influence on the decision boundary than the features with a narrower range of values. Therefore, for all features to have a similar influence in the construction of the decision boundary feature scaling is necessary.
 
### 5) Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?
The dual problem is faster to solve than the primal when the number of training instances is smaller than the number of features. So, in this case the primal form should be used.

### 6) Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ (gamma)? What about C?
As the model is underfitting, gamma and C should be increased.


