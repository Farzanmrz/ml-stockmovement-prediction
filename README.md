# Time-Series Stock Movement Classification: Simple and Ensemble Model Comparisons

This repository features code and the final report for time-series analysis of real-time stock data. This study was conducted as part of the final project that had to be delivered for the CS613: Intro to ML class at Drexel University. We attempted to compare different machine learning models and ensemble some of them to analyze which model was the best for predicting whether the stock price will go up or down in the next timestep

Status: <font color="green"> Completed </font>

## Project Intro/Objective
The project aimed to leverage real-time time-series stock market data to forecast stock movement in upcoming time intervals. We conducted a comprehensive analysis, comparing various machine learning models and exploring ensemble techniques to assess their accuracy in predicting stock direction. Diverging from conventional literature, which predominantly focused on inter-day intervals, our research delved into intra-day analysis, utilizing two months of 5-minute interval data for each stock. This study holds significance due to the profound influence of stock markets on society, with American stock market capitalization surpassing 45.5 trillion dollars in 2023, impacting global equity markets significantly. Our objective was to develop a more accurate method for stock price prediction, aligning with the needs of a substantial portion of the population invested in the stock market

### Methods Used  
* Intra-day Analysis: The project focused on analyzing intra-day intervals of 5 minutes, providing insights into stock market behavior at a finer temporal resolution
  
* Sector-Agnostic Approach: Instead of focusing on specific sectors, the study considered 66 stocks across all 11 sectors of the S&P 500, aiming to develop versatile models applicable to a wide range of stocks
* Understandable Machine Learning: In contrast to the prevalent use of complex and less interpretable models, this project prioritized the adoption of straightforward and transparent models, including Multiple Linear Regression, Logistic Regression, Linear Discriminant Analysis (LDA), Naive Bayes, and Decision Tree Learning. This choice aimed to underscore their proficiency in addressing high-frequency trading scenarios
* Ensemble Learning: Ensemble learning techniques were applied, incorporating weighted voting systems and RandomForest Classifiers, to enhance predictive accuracy
* Random Temporal Windows: Stock data was arranged by random days instead of chronologically to introduce necessary variability
* Data-Driven Model Selection: The research emphasized the importance of selecting the right model based on data characteristics, highlighting the impact of temporal resolution and the advantage of training models collectively on a diverse stock dataset.

### Technologies
* Python
* Jupyter
* Pandas, Numpy
* Conda

## Project Description
(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

## Needs of this project

- frontend developers
- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
- etc. (be as specific as possible)

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)


## Contributing DSWG Members

**Team Leads (Contacts) : [Full Name](https://github.com/[github handle])(@slackHandle)**

#### Other Members:

|Name     |  Slack Handle   | 
|---------|-----------------|
|[Full Name](https://github.com/[github handle])| @johnDoe        |
|[Full Name](https://github.com/[github handle]) |     @janeDoe    |

## Contact
* If you haven't joined the SF Brigade Slack, [you can do that here](http://c4sf.me/slack).  
* Our slack channel is `#datasci-projectname`
* Feel free to contact team leads with any questions or if you are interested in contributing!
