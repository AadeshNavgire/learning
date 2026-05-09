[Projects](#projects)
[Python](#python)
[Machine learning](#machine-learning)
[Generative AI](#generative-ai)


Projects

[TSP](#tsp)

[Content generation](#content-generative)

[ML based time series analysis](#ml-base-time-series-analysis)

[Data platform automation](#data-platform-automation)


## TSP: Technical sprint specification
This is a Multi-Agent Document Generation System built using LangGraph. The system takes an input file and automatically generates technical documents for SAP RICEFW objects like Report, Interface, Conversion, Enhancement, Form, and Workflow. We have 6 agents in the pipeline — Extraction, Group1, Group2, Unit Test, Reviewer, and Document Creation. After the Extraction agent runs first, the Group1, Group2, and Unit Test agents run in parallel to save time. All agents share a common state called DocumentState which acts like a shared memory — each agent reads from it and writes its output back to it. Once all three parallel agents finish, the Reviewer agent consolidates everything, and finally the Document Creation agent produces the final output. We also have an error handling mechanism where each agent writes a status flag like extraction_complete or extraction_failed, and downstream agents check these flags before proceeding. LangGraph gives us StateGraph to define nodes and edges, compile to build it, and invoke to run it — it handles all the parallel execution and state management internally.


## Machine learning

[Linear regression](#linear-regression)  [Neural network](#neural-network)
[Weights](#weights)[Bias](#bias)
[working on neural network](#working-of-neural-network)

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

### Activation Function
Activation function decides whether a neuron should be active or not.
It adds non‑linearity, helping the network learn complex patterns.
Without it, a neural network would behave like linear regression.

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
