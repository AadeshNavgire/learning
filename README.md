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


Linear regression: Is used to predict a continuous target variable based on a linear relationship between the target variable and one or more predictor variables. 

Aim is to find the best-fitting string line or hyperplan that minimize the difference between the predicted values and the actual value. 
