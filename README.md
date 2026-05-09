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

[Linear regression](#linear-regression)

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
