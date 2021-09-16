- #### **1) Try a Support Vector Machine regressor (sklearn.svm.SVR), with various hyper‐parameters such as kernel="linear" (with various values for the C hyperparameter) or kernel="rbf" (with various values for the C and gamma hyperparameters). Don’t worry about what these hyperparameters mean for now.How does the best SVR predictor perform?**

    The SVM Regressor was implemented with the mentioned hyperparameters and the results were compared with those of the other algorithm. Here is the **comparison between the RMSE** when both regressors were fit over the housing_prepared (X_train) and housing_labels (X_test) sets :
    * SVM = 70363.840
    * Random Forest = 18603.515
    * Difference : 51760.325
    
    Hence, from these results we can conclude that the Random Forest Regressor works far better than the SVM Regressor in this case. (Both linear and rbf kernels with C values ranging from 10-30000 and 1-1000 respectively and 0.01-3 gamma range for the rbf kernel. 50 candidates - 5 folds. **Best Params : {'C': 30000.0, 'kernel': 'linear'}**)
    
- #### **2)Try replacing GridSearchCV with RandomizedSearchCV.**
    The SVM regressor is now implemented using Randomized search.
  * RMSE = 54767.990
  
    We can conclude that randomized search tends to find better hyperparameters than grid search in the same amount of time. Both linear and rbf kernels with C are set as the reciprocals with shape parameters 20 and 200000 and a gamma with exponential distribution of scale 1.0.
    **Best Params : {'C': 157055.10989448498, 'gamma': 0.26497040005002437, 'kernel': 'rbf'}**)

- #### **3) Try adding a transformer in the preparation pipeline to select only the most important attributes**
        from sklearn.base import BaseEstimator, TransformerMixin

        def indices_of_top_k(arr, k):
            return np.sort(np.argpartition(np.array(arr), -k)[-k:])
        class TopFeatureSelector(BaseEstimator, TransformerMixin):
            def __init__(self, feature_importances, k):
                self.feature_importances = feature_importances
                self.k = k
            def fit(self, X, y=None):
                self.feature_indices_ = indices_of_top_k(self.feature_importances,self.k)
                return self
            def transform(self, X):
                return X[:, self.feature_indices_]
                
- #### **4) Try creating a single pipeline that does the full data preparation plus the final prediction**
        prepare_select_and_predict_pipeline = Pipeline([
        ('preparation', full_pipeline),
        ('feature_selection', TopFeatureSelector(feature_importances, k)),
        ('svm_reg', SVR(**rnd_search.best_params_))])
    
     * Predictions made by the pipeline: **Predictions:	 [203214.28978849 371846.88152572 173295.65441612  47328.3970888 ]
    Labels:		 [286600.0, 340600.0, 196900.0, 46300.0]**
        
    We can see that the pipeline's predictions are quite close to the the labels but these predictions would have been better if the best **RandomForestRegressor** model had been used instead of the best SVR regressor.

- #### **5) Automatically explore some preparation options using GridSearchCV**

        param_grid = [{
        'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
        'feature_selection__k': list(range(1, len(feature_importances) + 1))}]
    
        grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                        scoring='neg_mean_squared_error', verbose=2)
        grid_search_prep.fit(housing, housing_labels)
        
    Here in the param_grid, we have declared a list of strategies for imputing rather than just one like we did earlier. Along with the combinations of strategies we also have the feature_selection_k parameter that has been set to a list of integers ranging from 1 to the number of features or simply the length of our feature_importances list. These different combinations are performed by the GridSearch which other than param grid and Regressor have the same parameters. The result is stored in grid_search_prep which is later fit with the X_train and X_test sets, i.e, housing and housing_labels. The best parameter combination is as shown below :
    
    **grid_search_prep.best_params_
    {'feature_selection__k': 15, 'preparation__num__imputer__strategy': 'most_frequent'}**

    
  



