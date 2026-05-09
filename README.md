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

[Linear regression](#linear-regression)  			[Random forest](#random-forest)

[Neural network](#neural-network)

[Weights](#weights)[Bias](#bias)

[Working of a Neural Network](#working-of-a-neural-network)

[Activation function](#activation-function)

[Normal distribution/Gaussian distribution](#normal-distribution/gaussian-distribution)

[Bernoulli distribution](#bernoulli-distribution)

[Binomial distribution](#binomial-distribution)

[LSTM](#lstm)

[Entropy](#entropy)

[Loss function](#loss-function)

[Cost function](#cost-function)

[K-Nearest Neighbors](#k-nearest-neighbors)

[Random forest](#random-forest)

[Support Victor machine](#support-Victor-machine)

[Decision tree](#decision-tree)

[Naive Bayes](#naive-bayes)

[Global minima](#global-minima)

[Local minima](#local-minima)

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

Batch Gradient descent - It uses all data points to compute the gradient before updating the model. It accurate gradients. It can be slow and computationally expensive for large dataset. 
Stochastic Gradient descent(SGD) - It uses one data point a time to compute and update the gradient. It is faster for larger dataset, required less memory but noisy updates can make convergence slower or cause oscillations. 
