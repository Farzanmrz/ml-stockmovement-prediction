# Time-Series Stock Movement Classification: Simple and Ensemble Model Comparisons

This repository contains code and the final report for analyzing real-time stock data using time-series analysis. The study was conducted as part of the final project for the CS613: Intro to ML class at Drexel University.

## Objective

The project aimed to leverage real-time time-series stock market data to forecast stock movements in upcoming time intervals. We conducted a comprehensive analysis, comparing various machine learning models and exploring ensemble techniques to assess their accuracy in predicting stock direction.

## Methods Used

### Intra-day Analysis

The project focused on analyzing intra-day intervals of 5 minutes, providing insights into stock market behavior at a finer temporal resolution.

### Sector-Agnostic Approach

Instead of focusing on specific sectors, the study considered 66 stocks across all 11 sectors of the S&P 500, aiming to develop versatile models applicable to a wide range of stocks.

### Understandable Machine Learning

In contrast to the prevalent use of complex and less interpretable models, this project prioritized the adoption of straightforward and transparent models, including Multiple Linear Regression, Logistic Regression, Linear Discriminant Analysis (LDA), Naive Bayes, and Decision Tree Learning.

### Ensemble Learning

Ensemble learning techniques were applied, incorporating weighted voting systems and RandomForest Classifiers, to enhance predictive accuracy.

### Random Temporal Windows

Stock data was arranged by random days instead of chronologically to introduce necessary variability.

### Data-Driven Model Selection

The research emphasized the importance of selecting the right model based on data characteristics, highlighting the impact of temporal resolution and the advantage of training models collectively on a diverse stock dataset.

## Technologies

- Python
- Jupyter
- Pandas, Numpy
- Conda

## Project Description

Can we accurately predict stock market movements at a high temporal resolution (5-minute intervals) using interpretable machine learning models? Our hypothesis is that by focusing on the temporal granularity of intra-day data and employing interpretable models, we can achieve accurate predictions in high-frequency trading scenarios.

In contrast to the prevailing trend of utilizing complex, less interpretable models, we opt for a uniform modeling strategy that leverages the aforementioned models, prized for their clarity and computational efficiency, proving invaluable in navigating the dynamic landscape of financial markets.

With a dataset comprising 66 stocks across all 11 sectors of the S&P 500, our project strives to develop adaptable models capable of accommodating previously unseen stocks, thus departing from the customary practice of stock-specific modeling. Additionally, we employ ensemble learning techniques, incorporating weighted voting systems and RandomForest Classifiers.

Our analysis, based on classification metrics, unequivocally demonstrates the superiority of traditional linear models, with a particular emphasis on LDA, over more intricate ensemble methods within our experimental context. This discovery underscores the pivotal role of model selection based on data characteristics, highlighting the influence of temporal resolution, and accentuating the necessity of collective model training on a diversified stock dataset.

In essence, our research challenges the contemporary inclination toward advanced yet less transparent models, shedding light on the effectiveness of simpler, more interpretable models in the realm of high-frequency trading.

## Methodology

Our project adopts a multifaceted approach aimed at precise stock market movement prediction. This comprehensive strategy is structured into three core phases: data preprocessing, model training, and ensemble prediction. Building on recent advancements in stock market forecasting, our study delves into the intricacies of intra-day intervals, providing insights into market behavior nuances that go beyond conventional methods.

### Data Preprocessing

We curated a dataset with 66 stocks from various sectors of the S&P 500 using the yfinance module in python's conda environment to obtain data for a comprehensive analysis. We collected intra-day data at 5-minute intervals for each stock from October to December 2023. The data structure included key features like Open, High, Low, Close, Adjusted Close, Volume, 'window_ID,' 'timestep,' and 'volatility.' We prepared the dataset for training, testing, and unseen data by partitioning and normalizing it.

### Model Training

In this section, we present the supervised learning-based classification models employed in our study, each tailored to address the classification problem posed by our research.

- **Multiple Linear Regression:** Used to discern linear trends in stock market data, this model captures linear relationships between variables.

- **Logistic Regression:** Exceling in binary classification tasks, this model calculates the probability of binary outcomes.

- **Linear Discriminant Analysis (LDA):** Renowned for separating distinct classes effectively.

- **Naive Bayes Classifier:** Adapted for probabilistic classification, this model handles continuous features and assumes feature independence given the class label.

- **Decision Tree Classifier:** Specifically designed to model complex decision-making processes.

Our project combines interpretable machine learning models with data-driven strategies to address the research question effectively.

### Ensemble Learning

Ensemble learning is a pivotal aspect of our computational approach to stock market prediction. This technique involves the strategic combination of multiple machine learning models to enhance the overall predictive power and accuracy and mitigate the weaknesses inherent in individual models.

#### Simple Ensemble Learning with All Classifiers

This ensemble combines the strengths of all our classifiers using a weighted mean approach.

#### Simple Ensemble Learning with Best Classifiers

In this refined approach, we focus on a select group of high-performing classifiers.

#### Random Forest Ensemble Classifier

Our Random Forest model leverages multiple Decision Trees to improve the robustness and accuracy of stock market predictions.

Our project combines interpretable machine learning models with data-driven strategies, enhancing the reliability and performance of stock price movement prediction.


## Getting Started

1. Edit data_gathering.ipynb to set any date range, although 5-minute intervals can be obtained only for the past 2 months using yfinance, and generate the readable, original, ml usable csv files for testing  
2. All models are natively implemented in the models2.py file
3. Use prj_main.ipynb to print the comparative result of all models

## Contributing Members

* Farzan Mirza: farzan.mirza@drexel.edu (https://www.linkedin.com/in/farzan-mirza13/) 
* Nakul Narang
