[Projects](#projects)
[Python](#python)
[Machine learning](#machine-learning)
[Generative AI](#generative-ai)


## Projects

[TSP](#tsp)

[Content generation](#content-generative)

[ML based time series analysis](#ml-base-time-series-analysis)

[Data platform automation](#data-platform-automation)


## TSP: Technical sprint specification
This is a Multi-Agent Document Generation System built using LangGraph. The system takes an input file and automatically generates technical documents for SAP RICEFW objects like Report, Interface, Conversion, Enhancement, Form, and Workflow. We have 6 agents in the pipeline — Extraction, Group1, Group2, Unit Test, Reviewer, and Document Creation. After the Extraction agent runs first, the Group1, Group2, and Unit Test agents run in parallel to save time. All agents share a common state called DocumentState which acts like a shared memory — each agent reads from it and writes its output back to it. Once all three parallel agents finish, the Reviewer agent consolidates everything, and finally the Document Creation agent produces the final output. We also have an error handling mechanism where each agent writes a status flag like extraction_complete or extraction_failed, and downstream agents check these flags before proceeding. LangGraph gives us StateGraph to define nodes and edges, compile to build it, and invoke to run it — it handles all the parallel execution and state management internally.


## Machine learning

[Linear regression](#linear-regression)  	[Random forest](#random-forest) 	[Decision trees](#decision-trees)

[Neural network](#neural-network)		[AdaBoosting](#adaBoosting)		[classification](#classification)

[Weights](#weights)[Bias](#bias)	[Gradient boosting](#gradient-boosting)		[Logistic regression](#logistic-regression)

[Working of a Neural Network](#working-of-a-neural-network) 	[XGBoost](#xGBoost)		[Regression](#regression)

[Activation function](#activation-function)		[Gradient descent](#gradient-descent)		[Linear regression](#linear-regression)

[Normal distribution/Gaussian distribution](#normal-distribution/gaussian-distribution) 	[Polynomial regression](#polynomial-regression)

[Bernoulli distribution](#bernoulli-distribution) 	[Batch Gradient descent](#batch-gradient-descent)

[Binomial distribution](#binomial-distribution)		[Stochastic Gradient descent](#stochastic-gradient-descent)

[LSTM](#lstm) 	[Bagging](#bagging) 	[Regularization](#regularization)

[Entropy](#entropy) 	[Boosting](#boosting)		[Ridge Regression](#ridge-regression)

[Loss function](#loss-function) 	[Bias](#bias)		[Lasso Regression](#lasso-regression)

[Cost function](#cost-function)		[Variance](#variance)

[K-Nearest Neighbors](#k-nearest-neighbors) 	[Overfitting](#overfitting)

[Random forest](#random-forest)		[Underfitting](#underfitting)

[Support Victor machine](#support-Victor-machine)		[Cross validation](#cross-validation)

[Decision tree](#decision-tree)		[Covariance](#covariance)

[Naive Bayes](#naive-bayes)		[One-hot encoding](#one-hot-encoding)

[Global minima](#global-minima) 	[Linear Discremental analysis](#linear-discremental-analysis)

[Local minima](#local-minima) 		[Cosine similarity](#cosine-similarity)

[Accuracy](#accuracy) [Precision](precision) [Recall](#recall)  [F1 Score](#f1-score)

[Mean square error](#mean-square-error) [Mean absolute error](#mean-absolute-error) [Root mean square error](#root-mean-square-error) [R-Square](#r-square)

[Scaling](#scaling) [Normalization](#normalization) [Ensemble learning](#ensemble-learning)

[Clustering](#clustering) [Evaluation metrics in clustering](#evaluation-metrics-in-clustering)

[Principle component analysis](#principle-component-analysis) [Receiver operating character](#receiver-operating-character) [Area under curve](#Area-under-curve)

[Dropout](#dropout) [Backpropagation](#backpropagation) [Optimizers](#optimizers) [Batch normalization](#batch-normalization) [Time series](#time-series) [Prophet](#prophet) 

[Max pooling](#max-pooling) [Mean Pooling](#mean-pooling) [Min Pooling](#min-pooling) [Padding](#padding)

[Hyperparameter](#hyperparameter) [GAN](#gan) [VAE](#vae) [Latent space](#latent-space) [IQR method](#iqr-method)

[Diffusion Model vs GAN](#diffusion-model-vs-gan)


### Linear Regression
Is used to predict a continuous target variable based on a linear relationship between the target variable and one or more predictor variables.
The main aim is to find the best‑fitting straight line (or hyperplane) that minimizes the difference between the predicted values and the actual values.

Working:
Data Representation:
Each data point (for example, a house) is represented in a 2D space, where the size is on the x‑axis and the price is on the y‑axis.

Finding the Regression Line:
The algorithm determines the line that minimizes the sum of squared errors, commonly using the least squares method.

Equation of the Line:
Price=m×Size+c\text{Price} = m \times \text{Size} + cPrice=m×Size+c
where m is the slope and c is the intercept.

Prediction:
Once the line is learned, it can be used to predict the price of a new house based on its size.

Advantages:
Simple and easy to understand
Interpretable: The coefficients explain the relationship between variables.
Computationally efficient: Training and prediction are fast.

Limitations:
Assumes linearity: Not suitable for non‑linear relationships.
Sensitive to outliers: Extreme values can significantly affect the regression line.

### Neural network
A neural network is made of layers of connected nodes (neurons).
Each neuron does a simple calculation and passes the result to the next layer.

### Weights
Weights show how strong the connection is between two neurons.
Every connection has a weight.
During training, weights are changed to reduce errors between predicted and actual output.
Example:
If input is x and weight is w, then the weighted input is:
w × x

### Bias
Bias is an extra value added to the weighted input.
It helps the model fit the data better.
The total input becomes:
w × x + b
Bias allows the neuron to give output even when input is zero.

### Working of a Neural Network
1. Input Layer
Takes the input data and sends it to the next layer.

2. Hidden Layers
Each neuron:
Calculates: z = w × x + b
Applies activation function: a = activation(z)

3. Output Layer
Gives the final result, such as:
Classification (yes/no, cat/dog)
Regression (price, score, value)


### Activation Function
Activation function decides whether a neuron should be active or not.
It adds non‑linearity, helping the network learn complex patterns.
Without it, a neural network would behave like linear regression.

### Sigmoid
A smooth S-shaped function that maps inputs to a range between 0 & 1, use for probability estimation. 

<img width="80" height="32" alt="image" src="https://github.com/user-attachments/assets/d8ccf227-0f60-4ef4-91d2-6a0b7137edfe" />


Pros - Good for output representing probabilities. Well suited for binary classification problem.

Cons - Vanishing Gradient problem - Gradients become very small for large/small input slowing training. Output not centered around zero, which can affect convergence. 

When to use - In the output layer of the binary classification model. 
Rarely used in hidden layers due to the vanishing gradient problem. 


### tanh(Hyperbolic Tangent)
A sigmoid like function that maps inputs to a range between -1 and 1 centered at zero for better convergence. 

<img width="141" height="33" alt="image" src="https://github.com/user-attachments/assets/e6578bb4-8d04-4b99-835c-f8fdc3a37d69" />


Pros - Centered at zero which helps with faster convergence compared to sigmoid. Handles negative input better than sigmoid.

Cons - Sufferers from the vanishing gradient problem for large inputs, though less severe than sigmoid. 

When to use - In hidden layer when input may have a mean close to zero. 

### ReLU(Rectifier Linear Unit)
A simple piecewise linear function that output x if x> 0 and 0 otherwise introducing sparsity and computational efficiency. 

	f(x) = max(0,x)

Output range [0,)
Shape: Linear for x>0, zero otherwise. 

Pros - Simple and computationally efficient. 
Avoids vanishing gradient problem for positive inputs.
Prompts sparsity. 

Cons - Dying ReLU problem - Neurons can become permanently inactive for negative inputs during training, leading to dead neurons. 

When to use - Default activation function for hidden layer in most DL architecture. 
Works well for both convolutional and feedforward network. 

### Softmax
Converts a vector of logits into a probability distribution, where each output lies between 0 and 1 and sums to 1. 

<img width="82" height="41" alt="image" src="https://github.com/user-attachments/assets/d9196631-4b67-4f71-8f19-7889787ccf45" />

	
Pros: Suitable for multi class classification report. Output interpretable probabilities. 

Cons: Not used in hidden layers. Sensitive to large values in input logics (May require normalization) 

When to use: Output layer for multi-class classification models. 

The graphical representation is same as sigmodi, but softmax is use for multi-class classification. 

Neural network regression specifically refers to neural network designed for regression task, with modification in the output layer and loss function to handle continues output. 


### Normal distribution/Gaussian distribution
It is a probability distribution that describes how data is distributed in a symmetric. 

Application:  1) Gaussian Navie bays classifier. 
2) Linear regression
3) Data normalization 

### Bernoulli distribution
It is probability distribution which is used where out dataset contain yes, no or 0 or 1 or success or failure term. 

	p(success) = p
	p(failure) = 1-p

### Binomial distribution
It is a generalized distribution to multiple trials, it is a number of successes in a fixed number of independent trails. 

Bernouli distribution vs Binomial distribution

It has only one trail vs It has number of trails

Two outcome vs Number of success in n trails

One parameter p(probability of success) vs Two parameters n (number of trails) & p (probability of success) 

### LSTM 

(Long Short‑Term Memory) is a type of recurrent neural network (RNN) that can learn and remember information over long sequences.
It uses special gates to decide what information to remember, update, or forget, making it useful for tasks like time series, speech, and text prediction

Architecture of LSTM contain 3 main components - Cell state, Gates, Hidden state. 

<img width="464" height="239" alt="image" src="https://github.com/user-attachments/assets/6a59e0d0-6c81-4390-b07c-ccf823ee16d6" />

Working - 
Forget gate: The LSTM decides which part of the cell state. 
			Use sigmoid activation function.
Input gate: Determine which information should add to the cell state. 
		      A sigmoid activation function decide the importance of new information
		     A tanh generate candidate values for that state
Update gate: Determine the output at this time step
Hidden state: This state is update using output gate and cell state.

### Entropy
Entropy is a measure of uncertainty or impurity in a random variable.
It depends on how the probabilities are distributed.
Key Properties:

Maximum Entropy:
Entropy is highest when all outcomes have equal probability (maximum uncertainty).
Zero Entropy:
Entropy is zero when the outcome is completely certain, meaning there is no uncertainty.

Entropy is commonly used in decision trees to check how well the data is split.

Cross Entropy
Cross entropy measures the difference between two probability distributions:

The true distribution
The predicted distribution

It is mainly used as a loss function to measure how well a classification model is performing, especially in binary and multi‑class classification tasks.

Binary Cross Entropy
Binary cross entropy is used when there are only two classes.

It compares true labels with predicted probabilities.
Because it uses a logarithmic function, wrong predictions are penalized more, which helps the model improve accuracy during training.


Categorical Cross Entropy
Categorical cross entropy is an extension of binary cross entropy.

It is used for multi‑class classification problems.
It is applied when there are more than two classes.



### Loss function
It calculates the error for a single training example. It measures the difference between the predicted output and the actual target value for that specific example. 

Goal - To minimize the error on each individual prediction. 

### Cost function
It calculates the average loss over the entire training dataset. It sum of all the individual loss function values divided by the number of training example. 

Goal - To find the model parameters that minimize the overall error across the entire dataset. 



### K-Nearest Neighbors
KNN is a simple and non-parametric ML algorithm used for classification and regression tasks. It predicts the class or value of the data points based on its l closest neighbors in the feature space.
In simple language - Imaging you move into a new neighborhood and want to know what people like to eat. You ask your k-nearest neighbor to take the majority options. 
Feature of KNN - 1) KNN does not learn during the training, it stores the data and makes predictions during inference. 
2) Distance metric - Common once include euclidean. 
3) Choice of K - A small K can be noisy & large K smooth predictions but might overlook local patterns. 

### Naive Bayes
It assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. 
It is a probabilistic machine learning algorithm based on Bayes theorem. 

Example - You have a basket of fruits with an apple and orange with below characteristics. 
Color - Red, Orange, Green
Shape - Round, Oval
Texture - Smooth, Rough
Now, you have a new fruit that is red, round and smooth. A native bayes classifier would figure out if this is more likely an apple or an orange. 
It assumes the feature (color, shape, texture) are independent of each other. In reality, a red fruit might be slightly more likely to be round. But Navie bayes ignores this potential connection, keeping things simple. 

Naive bayes is a probabilisting classifier that is often employed when you have multiple or more than two classes in which you want to place your data. This algorithm is particularly used when you are dealing with text classification with a large dataset and many features. 

### Decision tree
It visually and explicitly represents decisions and their possible consequences. Each internal node of the tree represent a test on an attribute, each branch represent the outcome of the test and each leaf node represent a class label (For classification) or prediction value (for regression).
Advantages:-
Easy to understand and interpreter- The trees structure is usually intuitive 
It can handle both numerical and category data 
requires little data preparation and often needs less processing than other algorithms. 

Limitation:- 
Can overfit training data to poor generalisation to new data this can mitigate with pruning technique. 
sensitive to small changes in data can result in different trees and unstable predictions
can bias towards features with many categories. 

### Support Victor machine
SVM is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the optimal hyperplan that maximizes the margin between different classes in the dataset.
Is to find an optimal hyperplane that maximally separates data points into different classes.

Advantages: Effective in high- dimensional space work well even with many features. 
versatility can handle both linear and non linear data using different kernels.
memory efficiently uses only a subset of the training data in the decision function.

Limitations: Can be computationally expensive, training can be slow for large datasets.
Sensitive to the choice of kernel - Selecting the appropriate kernel can be crucial for performance. 
Not easily interpretable - The decision boundary can be complex, making it harder to understand the reasoning behind predictions. 

### Random forest
Its machine learning algorithm that operates by constructing a multiple of decision trees at training time and outputting the class that is the mean of the individual trees. 

Advantages - 1) High accuracy. 2) Robust to overfitting - The ensemble approach reduces the risk of overfitting to the training data. 3) Handle high dimensionality - can effectively deal with dataset with many features. 4) Handles high dimensionality - can effectively deal with dataset with many feature. 5) Provides feature importance - Can identify which feature are most important for prediction. 

Limitations - 1) Can be complex - The model can be difficult to interpret compared to a single decision tree. 2)  Can be computationally expensive - Training and prediction can take longer than simpler model. 3) Requires careful tuning - Parameter like the number of trees and features need to be optimized for best performance. 

### K Nearest Neighbors
It classifies data points based on how its neighbors are classified ‘k’ representing the number of nearest neighbors. 

Working - 1) Calculate distance - KNN calculates the distance between the new data point and all other data point in the dataset. Common distance measure include euclidean distance. 
2) Find nearest neighbors - It identifies the K nearest data points to the new data point based on the calculated distance. 3) Classify - In classification the algorithm assign the new points to the class that is most frequent among its nearest neighbors. 

Advantages - 1) Simple and easy to understand. The concept is straightforward and easy to implement. 2) No training period- KNN is a lazy learner, meaning it doesn't explicitly learn a model from the training data. It simply memories the data and uses it for prediction. 3) Versatile - Can be used for classification and regression. 4) Non-parametric - Makes no assumption about the underlying data distribution. 

Limitation - 1) Computationally expansive - Calculating distance for all data points can be time consuming especially for large dataset. 2) Sensitive to irrelevant feature - Including irrelevant feature can negatively impact performance feature selection is crucial. 3) Curse of dimensionality - Performance can degrade in high dimensional space. 


### Global minima
It is the absolute lowest point of a loss function, where a ML model achieves optimal performance. 

### Local minima
A point where the function value is lower than all nearby points, but may not be the lowest overall. 
Global minima - The lowest point of the entire function across its entire domain. 
Local maxima - A point where the function value is higher than all nearby points but may not be the highest overall. 
Global maxima - The highest point of the entire function across its entire domain. 
In Optimization algorithms (gradient descent) often get stuck at local minima instead of reaching the global minimum, which is why initialization and learning rate matter. 



### Random forest
Is a method that uses bagging with decision trees. It builds multiple decision trees on different subsets of the data and combines their prediction (by averaging) to improve accuracy and reduce overfitting. 

In simple language - Its like asking multiple experts for advice and taking the average of their opinions to make a decision. 

Working:- 1. Bootstrap aggregating - Random forest creates multiple subset of the original data through random sampling with replacement. 
2. Feature randomness - For each tree, a random subset of feature is selected to determine the best split at each node. 
3. Building - Each subset is used to build a decision tree, typically without pruning. 
4. Aggregation - The predictions from al the tree are combined to produce the final prediction. 

### AdaBoosting
(Adaptive boosting)Adaboost combines multiple weak learners sequentially. Each new learner focus on correcting the error made by the previous once and weights are assigned to prioritize difficult. 
Example - 1) Start with an easy question. 2) Focus on the questions the student got wrong in the next lesson. 3) Repeat unit the student performs well on all questions. 

### Gradient boosting
It builds model sequentially with each new model minimizing the residual error of the previous model. It uses gradient descent to optimize the loss function. 
Use gradient boosting where accuracy is critical and you have a moderate size dataset with complex patterns. 

### XGBoost
Extreme Gradient boosting - XGBoost is an optimized implementation of the gradient boosting algorithm designed to be fast efficient and scalable. 

### Gradient descent
Is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent as defined by the negative of the gradient. 
 It is use to find out the best solution or optimal parameter of a model. 
Its a concept of a global minima where loss is minimum. 
Learning rate is the steps to find out the global minima. 
Example - You are standing at the top of the hill. You want to reach the lowest point (global minimum). So, you will take small steps, step by step, to reach the lowest point (learning rate). This entire process is called Gradient Descent.

### Batch Gradient descent
It uses all data points to compute the gradient before updating the model. It accurate gradients. It can be slow and computationally expensive for large dataset. 

### Stochastic Gradient descent
(SGD) It uses one data point a time to compute and update the gradient. It is faster for larger dataset, required less memory but noisy updates can make convergence slower or cause oscillations. 

### Bagging
(Bootstrap aggregation) Bagging is a technique where multiple models are trained on different random parts of the same dataset, and their results are combined by averaging or voting. 

Bagging is used when high variance models or large datasets with sufficient computational power. 

Bagging models work independently and their results are combined at the end. 

### Boosting
The models are trained one after another and each new model tries to fix the mistake of the previous once. The results are combined to create a strong final model. 

Boosting is used where data with complex patterns or when you need high accuracy or cases where bias needs to be reduced (Underfitting problem). 

Boosting models work in a sequence learning from each other's mistakes. 


### Bias
Is the error introduced by approximating a real-world problem, which may be too complex with a simplified model. 
Ex- High bias (underfitting) 
Predicting student grades based solely on attendance. 
Mode - A simple linear regression. 
The model predicts all grades too similarly because it ignores other factors like study habit or exam difficulty. 

### Variance
Variance refer to the models sensitivity to small changes in the training data. High variance implies that the model captures noise in the data, leading to overfitting. 
Ex- High variance 
Predicting student grades based on attendance, mood, time spent on social media. 
Model - A high- degree polynomial regression. 
The model fits the training data perfectly but makes poor predictions for new students because it learns irrelevant patterns. 

### Bias-Variance tradeoff
The bias-variance tradeoff is the balance between bias and variance to minimize the total error on the model. The goal is to create a model that captures patterns in the data without overfitting or underfitting. 



### Overfitting
It occurs when a model learns the training data too well, including noise and random fluctuations, rather than generalizing the underlying patterns. As result it performs well on the training data but poorly on unseen test data. 

Example - Imagine you are trying to predict tomorrow's weather using past weather data. Overfitting is like memorizing each day's weather from the past rather than understanding general weather patterns. This makes your prediction only work well for specific days but fail for new, unseen days. 

How to avoid overfitting?
Regularization - Add penalties to large weights using L1(Lasso) and L2(Ridge) regularization. 
Use cross validation to monitor model performance on unseen data. 

### Underfitting
When the model is too simple to capture the underlying structure of the data. It performs poorly on both the training data and test data because it fails to learn the patterns in the data. 

Example - Underfitting is like always predicting ‘sunny’ weather no matter the actual patterns in the data. This overly simple approach fails to capture the nuance of changing weather. 

How to avoid underfitting?
Switch to a model capable of capturing more patterns (e.g - move from linear regression to polynomial regression) 

### One-hot encoding
A method to represent a categorical variable as binary vector, where each category is represented as one-hot vector. 

It is used when working with categorical data in model that required numerical input such as logistic regression. 

### Covariance
It is a statistical method used to find the relationship between two random variables. 
Cov>0 - Positive: If cov is positive means that one value is increasing wrt another value
Cov<0 - Negative: If cov is negative means that one value is increasing and another value is decreasing 
Cov=0 - Zero: There is no relation between two variables. 

cov(x,y) = 1N(xi-x)(yi-y)
Use: 
Dimensionality reduction (PCA)
Feature selection (Regression) 
Classification (LDA, Naive bayes) 
Clustering (Gaussian mixture model) 
Modeling uncertainty(time series) 

### Cross validation
It is used to assess the performance of a model by dividing the dataset into multiple subset and testing the models ability to generalize the unseen data. 

> Instate of using single train-test split, cross validation will help the model to perform well on different subset of the data, it reduces the risk of overfitting or underfitting. 

How cross validation work.
> The dataset is split into k subset (folds) 
> One dataset is used as the validation set and the remaining k-1 subset are used as the training set. 
> The model is trained on k-1 subset and evaluated on the remaining fold. 
> This process is repeated k times, with each fold being used as the validation set exactly once. 
> The performance metric (accuracy, F1) is averaged across all k folds to provide a more reliable estimate of the model performance. 

To compare or select best model or hyperparameter use cross validation. 

### Linear Discremental analysis
(LDA)Is use to find a linear combination of feature that best separate two or more classes of data. 
Example- Imagine you have data with different classes (like different types of animals: cats, dogs, and rabbits), and you want to find a way to separate these animals based on their features (like weight, height, and fur length). LDA will help you project these features into a new space where the separation between the classes is as clear as possible, making it easier to classify new data points.

### Cosine similarity
It calculates the cosine angle between two vector, which tells how similar they are based on direction and their magnitude. 

		cosine similarity = A.B||A|| ||B||

### Decision trees
A tree like model that splits data based on features values to make decision.

### classification
Classification is to predict a discrete class label for given input data. 

### Logistic regression
It predicts the probability of a binary class using the logistic function, such as between 0 and 1, converted to class label (Yes/No). 


### Regression
Regression is a type of supervised learning which is used to predict a continuous output variable based on one or more input features. 

### Linear regression
A linear relationship between the input feature and the target variable. 
Use linear regression for simple linear relationship. 

### Polynomial regression
Extends linear regression by fitting a polynomial curve to capture non-linear relationships. 
Use polynomial regression for non-linear relationships. 

### Regularization
It helps prevent the model from overfitting the training data, so it can perform better on new data. 
Its like a gentle rule that tells the model "don't make things too complicated”

### Ridge Regression
Shrinks coefficient towards zero but not zero.
It keep all features but shrkins coefficient
Multicolinearily, when all features are useful
A type of linear regression that includes an L2 regularization term to reduce overfitting. 
Use ridged, lasso for regularization to avoid the overfitting of model. 


### Lasso Regression
It adds L1 regularization which can shrink some coefficient to zero this is more effectively performing feature selection. 
For dimensionality use lasso regression for feature selection in high-dimensional dataset. 
It performs automatic feature selection

### Cosine similarity
It calculates the cosine angle between two vector, which tells how similar they are based on direction and their magnitude. 

cosine similarity = A.B \ ||A|| ||B||


### Accuracy
The correctly classified instance to the total instance. 

	Accuracy = True positive (TP) + True negative (TN)\Total Instance

Accuracy is used when the dataset is balanced. I.e. the classes have similar proportions. 
Example - Predicting student pass/ failed outcomes in an evenly distributed dataset. 

### Precision
The proportion of correctly predicted positive observations to the total predicted positive observations. 

	Precision = True Positive (TP)\True positive (TP) + False positive(FP)

It is use when false positives are costly and in email spam detection, falsely making a legitimate email as spam is undesirable. 

### Recall
The proportion of correctly predicted positive observations to all actual positive observations. 

	Recall = True positive (TP)\True positive (TP) + False negative (FN)

It use when false negative are costly or critical to avoid. In medical diagnosis, missing a disease case (false negative) can be life threatening. 

### F1 Score
It is harmonic meaning od precision and recall. 

	F1 score = 2Precision X Recall\Precision + Recall = TP\TP + 1\2(FP+FN)

It is use mostly when you need to balance between precision and recall, especially for imbalance dataset.


### Mean square error
(MSE) - It averages the squared difference between predicted and actual value. 
		
<img width="98" height="38" alt="image" src="https://github.com/user-attachments/assets/429b5d6b-c6fe-4798-a5a7-39b3d7f58494" />

### Mean absolute error
(MAE) - It averages the absolute difference between predicted and actual value. 

<img width="87" height="41" alt="image" src="https://github.com/user-attachments/assets/fad6e8ab-aea8-4e10-b6aa-bd4eff7e7730" />

		
### Root mean square error
(RMS) - It measures the average magnitude of the errors between predicted and actual values.
	
<img width="130" height="50" alt="image" src="https://github.com/user-attachments/assets/93b71f68-416d-414f-949b-79cf87463a14" />


### R-Square
R2 - Co-efficient of determination (COD) - It represent the goodness of fit of a regression model. The R value lies between 0 and 1. Where 1 represent when model perfectly fits the data and there is no difference between predicted and actual value. Where 0 represent the model does not predict any variable in the model and it does not learn any relationship between the dependant and independent variable. 

<img width="212" height="56" alt="image" src="https://github.com/user-attachments/assets/67e1b8d9-3eda-4e50-9487-6eb898fa417d" />



### Scaling
The scaling is the process of transforming data into a specific range or unit. 
It is important when features in the dataset have different unit or magnitude. 
Ex- one feature might be measured in meters, while another is in kilogram. 

Types of scaling
1. Standardization(z-score normalization) - It transforms the data to have a mean of 0 and standard deviation of 1. 
Impact - Is used for models like linear regression, logistic regression, support vector machine (SVM) and principle component analysis (PCA). 

2. Min-max scaling (Normalization) - It rescales the data to a fixed range, usually [0,1]
Min-max scaling is sensitive to outliner because outliner will stretch the feature range leading to distorted transformation. 
It is widely used in KNN, neural network and Gradient descent models. 


### Normalization
Adjusting the value in a dataset to fit a specific range. Typically [0,1]



### Ensemble learning
Is based on the idea that multiple diverse models working together can reduce errors, improve predictions and increase the robustness of the model. 
By combining multiple models, ensembles often outperform individual models, especially when the base learners are weak. 
The variance of the model is reduced by averaging predictions, leading to better generalization on new data. 
Ensemble models tend to be more robust to noise and outliners than individual models. 
Ensemble methods can help balance between the trade off between bias and variance. 




### Clustering
Is a unsupervised learning technique in machine learning in which data points are grouped together based on similarity distance between them. 
Working:
Find the number od cluster(K)
Choose k centroid from the data points randomly 
Compute the distance between centroid and each data point. 
Group into k cluster based on the minimum distance between the data points and centroid. 
Repeat step 3 and 4 until convergence by updating all points. 

Types of clustering: 
Partitioning methods - It divides the data into k clusters directly. It works by assigning data points to the nearest cluster center, iteratively refining the center.
	Example - K-Means
Hierarchical clustering - Builds a hierarchy of cluster using a tree like structure. It works by recursively merge or split clusters based on similarity. 
	Example - Agglomerative(Bottom-up), Divisive(Top-down) 

Density based method - Identify cluster as region of high-density points. It works by forming cluster where data points are close and leaves out noise or outliers. 

Model based methods - Assumes data is generated from a mixture of distributions and use statistical models to find cluster. It works by fitting the data to multiple probability distribution and assign points based on likelihood. 
	Example - Gaussian mixture models

Graph based method - Use graph theory to find cluster by treating data points as nodes in a graph. 
	Example - Spectral clustering




### Evaluation metrics in clustering 
Silhouette score - Measures how well each data point fits within its cluster compared to other cluster. It ranges -1 (poor clustering) to +1 (good clustering) 

<img width="109" height="36" alt="image" src="https://github.com/user-attachments/assets/c768dd1b-3032-41d2-a062-b2af5b61c712" />


Inertia - Measures the compactness of cluster by calculating the sum of squared distance between each point and its cluster center. 

<img width="152" height="37" alt="image" src="https://github.com/user-attachments/assets/fed95226-033e-490c-a677-2fc5b9f712cd" />


Adjusted Rand Index (ARI) - It measures the similarity between true labels and predicted clusters, adjusting of chance.

		ARI = Index - Expected Index / Max Index - Expected Index

Normalized mutual Information (NMI) - It measures how much information is shared between true labels and predicted clusters, normalized to account for varying cluster size. 

<img width="98" height="26" alt="image" src="https://github.com/user-attachments/assets/9bc78330-647d-4005-99bb-3746b4e40346" />



Purity - It measures how pure the clusters are by assigning each cluster to the class with the majority of points. 

<img width="162" height="35" alt="image" src="https://github.com/user-attachments/assets/2e5dc09a-926c-4bc9-af26-d05e290b807a" />


### Principle component analysis
Is is use for dimensionality reduction. To simplify complex dataset into a smaller set of variables. 
It helps us to remove number of feature from dataset while trying to keep the important patters and structure. 

### Receiver operating character
ROC Is a graphical representation used to evaluate the performance of binary classification model to show sensitivity (TPR - True positive rate) and specificity (False positive rate) across different threshold value. 


		TPR - True positive\True positive + False negative 
		
		FPR - False positive\False positive + True negative

### Area under curve
AUC Is a single value that summarizes the performance of the classifier. It represents the probability that the classifier will rank. 
AUC = 1 -> Perfect model 
AUC = 0.5 -> Random classifier 
AUC < 0.5 -> Worst model



### Dropout
Is a regularization used in NN to prevent overfitting. It make model more generalizable on unseen data by randomly “dropping out” a certain percentage of neuron during training(20-50%)
Work- Imagine you have a neural network with 5 neurons in a layer. During training you have decided to use a dropout rate of 0.4 which means 40%. 
In one training step neuron 1 and 4 might be dropped out and neuron 2,3,5 will be active. 
In another tanning set neuron 2,3 will dropped off and neuron 1,4,5 will be active. 
So this process makes the network learns to be robust and not overly depend on any particular neuron. 



### Backpropagation
It is a process in neural network which learns from its mistake and adjust internal parameter (weights and bias) to improve the performance. Its main agenda is to reduce error. 

### Optimizers
It is method use in neural network to adjust weight and biases of neural network during training to minimize the loss function. It use key component of gradient descent process. 

Analogy between backpropagation and Optimizers:- Backpropagation is like a detective that investigate why the network made a mistake by computing gradients. 
Optimizers is like the decision makers that decide how to fix the mistake by updating parameters. 

### Batch normalization
It normalize the inputs of each layer in neural network by adjusting and scaling activation. 

### Time series
A time series is a sequence of data points collected at regular intervals over time, where order matters. 

Data points are ordered chronologically -> Each points depends on previous values -> Exhibihts patterns: trends


### Prophet
Prophet is a time series forecasting library by facebook that’s excellent for business time series data. 
Prophet = Trend + Seasonality + Holiday Effects + Error

Seasonality in Prophet: Seasonality is repeating patterns at fixed intervals. 
Types of seasonality:
Yearly seasonality - Pattern repeats every year (sales spike in Dec). If data >= 2 years
Weekly seasonality - Patterns repeats every week (higher traffic on weekends) if data >= 2 weeks
Daily seasonality - Pattern repeats every day (more call during business hours) if data >= 2 days


### Dropout
It is a regularization technique used in deep learning to reduce overfitting by randomly deactivating a fraction of neurons during training. It prevents the network from becoming overly reliant on specific neurons and encourages more robust feature learning. 


### Max pooling
This operation selects the maximum value from the region of the feature map covered by the filter. It is useful when the goal is to identify the most prominent features, such as edges, textures. 

### Mean Pooling
This operation computes the average value of the elements in the region of the feature map covered by the filter. It provides a more generalized representation of the input and is useful when preserving the overall context is important. 

### Min Pooling
This operations selects the minimum value from the region of the feature map covered by the filter. It is useful when the goal is to identify the least prominent features. 

### Padding
Is the process of adding extra values around data to control the shape of inputs and preserver important information during processing. It is used to preserve the spatial dimensions of the input image after convolution operations on feature map. 
Valid padding: No padding is added, output size decrease after convolution. 
Same padding: Padding is added so that output size matches input size. 
Zero padding: Most common, where zeros are added round the boarded. 


### Hyperparameter
It is essential for defining the behavior and performance of ML modes. By casefully tuning these hyperparameters, we can optimize their model by preventing overfitting, underfitting and achieve better predictive performance. 

### GAN
GAN consist of two competing neural networks in an adversarial framework. It work on two neural networks as generator and discriminator. 
Generator: Creates fake data samples from random noise. 
Discriminator: Evaluates if samples are real or fake. Both improve iteratively through completion. 
Result: High-quality realistic synthetic data generation. 
Equilibrium: Ideally reaches state where D can’t distinguish. 


### VAE
A variational autoencoder is a generative models that learns a probabilistic latent space to generate new data. 
It uses an encore to map input data into a latent distribution defined by mean and variance, and a decoder to reconstruct data from sample latent vectors. 
VAEs are trained using a combined loss function consisting of reconstruction loss and KL divergence which regularizes the latent space. 
This enables smooth interpolation and controlled data generation. 

### Latent space
Is a compressed, continuous representation of data where each point encodes meaningful features learned by a generative model to create new sample.
Latent space: Enables generation of new, realistic samples. Captures semantic structure of data. Allows smooth interpolation between outputs. Supports controlled generation. Improves generalization and diversity. 



### IQR method
A statistical measure that helps to understand the spread of data by forcing on the middle 50% of a dataset. It calculated as the difference between third quartile (Q3) and the first quartile (Q1) using IQR = Q3-Q1

### Diffusion Model vs GAN

| Aspect | Diffusion Model | GAN |
|------|-----------------|-----|
| Data Generation Method | Generate data by gradually removing noise | Generate data via adversarial competition |
| Architecture | Single denoising network | Generator + Discriminator |
| Output Quality | High realism with good diversity | Sharp outputs but less diverse |
| Generation Speed | Slower | Faster |
