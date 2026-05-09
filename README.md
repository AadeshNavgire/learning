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

[Linear regression](#linear-regression)  

[Neural network](#neural-network)

[Weights](#weights)[Bias](#bias)

[Working of a Neural Network](#working-of-a-neural-network)

[Activation function](#activation-function)

[Normal distribution/Gaussian distribution](#normal-distribution/gaussian-distribution)

[Bernoulli distribution](#bernoulli-distribution)

[Binomial distribution](#binomial-distribution)

[LSTM](#lstm)


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

### LSTM (Long Short‑Term Memory) 
is a type of recurrent neural network (RNN) that can learn and remember information over long sequences.
It uses special gates to decide what information to remember, update, or forget, making it useful for tasks like time series, speech, and text prediction
