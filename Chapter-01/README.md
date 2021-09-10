- #### **1) How would you define Machine Learning?**
    * Machine learning is the science of making machines / computer programs learn from data  without having to code explicitly.
    * A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

- #### **2) Can you name four types of problems where it shines?**
    * Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better.
    * Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution.
    * Fluctuating environments: a Machine Learning system can adapt to new data.
    * Getting insights about complex problems and large amounts of data
- #### **3) What is a labeled training set?**
    It is the training data that includes the desired solutions (labels) for identifying certain properties.
- #### **4) What are the two most common supervised tasks?**
    * **Classification** -  Example - Spam filter, it is trained with many example emails along with their class (spam or ham) and then it learns to classify new emails.
    * **Regression** - It is done to predict a target numeric value, such as the price of a house using a given set of features (bedrooms, per square feet area, location, etc) called predictors. To train the system many examples of cars, including both their predictors and their labels (prices of the houses) must be fed to the regression algorithm.
   
 - #### **5) Can you name four common unsupervised tasks?**

    * **Clustering algorithm**-  It involves grouping the data on the basis of attributes
    * **Visualization** - A lot of complex and unlabeled data is fed and the output is either 2d or 3d representation of data to identify unsuspected patterns.
    * **Dimensionality reduction** - Simplifying the data without losing too much information. one way to do this is by merging highly correlated features into one.
    * **Anomaly detection** - Detecting unusual credit card transactions, automatically removing outliers from a dataset, etc.
    * **Novelty detection** - The identification of new or unknown data or signals that a machine learning system is not aware of during training.
    * **Association Rule learning**- The goal is to dig into large amounts of data and discover interesting relations between the attributes.
    
- #### **6) What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains? (Pg - 15)**

    Reinforcement learning 

- #### **7) What type of algorithm would you use to segment your customers into multiple groups?**
     Clustering algorithm
- #### **8) Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem? (pg -5)**

    Spam detection is a supervised learning problem. It comes under the classification problem. In the training phase,  set of labeled training data of spam and ham are fed into the algorithm which learns from the this data about the words and phrases that are good predictors of spam by detecting unusually frequent patterns of words in spam examples compared to the ham examples and then tries to classify unknown emails into spam or ham.

- #### **9) What is an online learning system? (Pg - 16, 17)**

    In online learning, the system is trained incrementally by feeding it data instances sequentially, either individually or by small groups called mini batches. Each learning step is fast and cheap so that the system can learn about the new data as it arrives. 
    It is great for systems that receive data as a continuous flow(stock prices)
    and need to adapt to change rapidly. Once the system learns about new instances, that data can be discarded. In this way we can work even with limited resources and save huge amount of space.
    
-   #### **10) What is out-of-core learning? (pg - 17)**

    Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine’s main memory (this is called out-of-core learning). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.
- #### **11) What type of learning algorithm relies on a similarity measure to make predictions? pg - 18**

    Instance based learning algorithm. For example, In a spam filter, instead of flagging emails that are identical to known spam emails, filter could be programmed to also flag emails that are very similar to known spam emails. this requires a measure of similarity between two emails. It could be the count of words they have in common. A new email would be flagged as spam if it has many words in common with the known spam email.
    
- ####   **12) What is the difference between a model parameter and a learning algorithm’s hyperparameter? (pg - 30)**

    *Hyperparameter* is a parameter of a learning algorithm (not of the model). As such, it is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training. It controls the amount of regularization. If the regularization hyperparameter is set to a very large value then we will get an almost flat model(slope close to zero).
    
    *Model parameter* is a variable of the selected model which can be estimated by fitting the given data to the model. These are used to minimize the cost function.
 - #### **13) What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make precictions? (P21)**

    In Model-Based learning algorithms, the common strategy/process is as follows : 
    * First the model parameters are defined. (Θ)
    * Then, in order to find the optimal values for these parameters, the system is provided with a performance measure (*utility function/fitness function*) that measures how *good* the model performs or a cost function to measure how *bad* it performs with the data.
    * This process of *model training* is where the algorithm searches for the optimal parameter values for the model and then goes on to make optimal predictions with the same.
    
- #### **15) If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions? (P28)**

    This is the case of Overfitting. This occurs when the training dataset is too noisy or when it is insufficient. The model detects patterns in the noise itself which do not generalize to new instances.
    **Possible Solutions**
    * To simplify the model by selecting one with fewer parameters
    (e.g., a linear model rather than a high-degree polynomial
    model), by reducing the number of attributes in the training
    data or by constraining the model
    * To gather more training data
    * To reduce the noise in the training data (e.g., fix data errors
    and remove outliers)
    
- #### **16) What is a test set and why would you want to use it? (P31)**
    The whole data is split into the training set and the test set. The model is trained using the training set and is tested using the test set. The error on the new cases is called the *generalization error* or out-of-sample-error. By evaluating the model on test set we can get an estimate of this error. This value tells us how well the model will perform on new instances.

- #### **17) What is the purpose of a validation set?(P32)**
    * A part of the training set is held out to evaluate several candidate models and select the best among them.
    * The heldout set is called *validation set* or *development set* or *dev set*.
    * Multiple models are trained with various hyperparameters on the reduced training set (i.e., the full training set minus the validation set), and the model that performs best on the validation set is selected. After this holdout validation process, the best model is trained on the full training set (including the validation set), and this gives the final model. Lastly, this final modelis evaluated on the test set to get an estimate of the generalization error.
- #### **18) What can go wrong if you tune hyperparameters using the test set?(P32)**

    If hyperparameters are tuned using the test set, the final produced model will have a higher percentage of generalization error. This is beacause generalization error will be measured multiple times on the test set.
    
- #### **19)What is repeated cross-validation and why would you prefer it to using a single validation set?(P32)**

    A very small or very large validation set is not ideal for evaluation and might lead to selection of suboptimal models. One solution to this problem is to perform repeated *cross validation*, using several small validation sets instead of one. Each model is evaluated once per validation set, after it is trained on the rest of the data. By averaging out all the evaluations of a model, we get a much more accurate measure of its performance.