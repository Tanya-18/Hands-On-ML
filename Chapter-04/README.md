 
- ### 1) What Linear Regression training algorithm can you use if you have a training set with millions of features?
    - **Stochastic Gradient Descent** or **Mini Batch Gradient Descent** algorithm can be used for a training set with millions of features.
    - ***Stochastic Gradient Descent*** picks a random instance in the training set at every step and computes the gradients based only on that single instance, which makes the algorithm faster and only one instance needs to be in memory at each iteration.But due to its stochasticity the cost function bounces up and down instead of decreasing gradually, the only advantage here is that it will cross the local minima.The learning rate has to be reduced gradually.
    - ***Mini Batch Gradient Descent*** computes gradients on small random sets of instances called mini batches. Its advantage over SGD is that we get a performance boost from hardware optimization of matrix operations. But it is harder for mini batch GD to escape from local minima.
    - Both these algorithms can work really well with a good learning schedule.

- ### 2) Suppose the features in your training set have very different scales. What algorithms might suffer from this, and how? What can you do about it?
    - The Gradient Descent algorithms will suffer from this.Due to different scales of the features the algorithm will take a long time to converge towards the minimum as the cost function here will be an elongated bowl. **Feature Scaling** is to be done to avoid this problem.
        
- ### 3) Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?
    No, because the cost function of Gradient Descent is **Convex**.When we join any two points on the curve the line segment joining them never crosses the curve which means there is no local minima and just one global minimum.
        
- ### 4) Do all Gradient Descent algorithms lead to the same model provided you let them run long enough?
    - All the Gradient Descent algorithms end up with very similar models and make predictions in exactly the same way.
    - All these algorithms will end up near the minimum, but Batch GD's Path will stop at the minimum while both Stochastic and Mini-Batch GD continue to walk around. Both these algorithms will also reach the minimum provided that a good learning schedule is used.
    
- ### 5) Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?
    If the validation error consistently goes up after every epoch, and if the training error also goes up then the learning rate is too high due to which the algorithm is diverging.
    To fix this we need to find a good learning rate using grid search. However, if the training error is not going up, then the model is overfitting and we need to regularize it.

- ### 6) Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?
    In Mini-batch Gradient Descent the curves are not so smooth which makes it difficult for us to realize if we have reached the minimum or not. So it would be better to stop only after the validation error has been above the minimum for some time.

- ### 7) Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?
    Mini Batch Gradient Descent will reach the optimal solution faster.
    Batch Gradient Descent will converge at the minimum and stop but Stochastic GD and Mini-Batch GD will continue to walk around the minimum. But they will also reach the minimum if a good learning schedule is used.

- ### 8) Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?
    When there's a large gap between the training error and validation error it means that the model is performing better on the training data than on the validation data, that is the model is overfitting.
        * One way to improve an overfitting model is to feed it **more training data** until the validation error reaches the training error.
        * **Regularize the model** : Reduce the degrees of freedom.
        * **Early Stopping** - Stop Training as soon as the validation error reaches a minimum. Because after a while the validation error stops decreasing and starts to go back up which indicates that the model has started to overfit.

- ### 9) Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?
    * As training error and validation error both are high the model is underfitting which means it has high bias. The regularization parameter should be decreased to reduce bias.
    
- ### 10) Why would you want to use:
    • **Ridge Regression instead of plain Linear Regression (i.e., without any regularization)?**
    * In Ridge Regression a regularization term (using the l1 norm) is added to the cost function to avoid overfitting. It is more useful when there is multicollinearity in the features (when there are high correlations between two or more predictor variables. In other words, one predictor variable can be used to predict the other. This creates redundant information, skewing the results in a regression model.)
        
    • **Lasso instead of Ridge Regression?**
    * When only a few features are actually useful Lasso regression should be preferred over ridge as it reduces the weights of useless features down to zero.
    
    • **Elastic Net instead of Lasso?**
    * Elastic Net is preferred over Lasoo when the number of features is greater than the number of training instances or when several features are strongly correlated.
        
- ### 11) Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or one Softmax Regression classifier?
    Softmax Regression Classifier predicts only one class at a time. Since we need to output two classes (outdoor/indoor and daytime/nighttime) we will have to use two Logistic Regression classifiers.

        

        
        
        
    
       
  
   
   
