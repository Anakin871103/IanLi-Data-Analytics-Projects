# Project A: Customer Churn Prediction

This project aims to predict customer churn using various machine learning models. The goal is to identify customers who are likely to leave the service, allowing the business to take proactive measures to retain them.

## Overview

Customer churn prediction is a critical task for businesses that rely on subscription models. By understanding the factors that contribute to churn, companies can implement strategies to improve customer satisfaction and retention.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/IanLi-Data-Analytics-Projects.git
   ```

2. Navigate to the project directory:
   ```
   cd IanLi-Data-Analytics-Projects/projects/Project_A_Customer_Churn_Prediction
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**: The raw customer data can be found in the `data/` directory. Ensure that the data is preprocessed before training the models.

2. **Model Training**: The model training script is located in the `src/` directory. You can run the training script to train the churn prediction model:
   ```
   python src/churn_model.py
   ```

3. **Analysis**: The Jupyter notebook `Customer_Churn_Analysis.ipynb` in the `notebook/` directory contains the exploratory data analysis and model evaluation. You can open this notebook to visualize the results and insights.

## Project Structure

```
Project_A_Customer_Churn_Prediction/
├── README.md
├── notebook/
│   └── Customer_Churn_Analysis.ipynb
├── src/
│   └── churn_model.py
└── data/
    └── customer_data.csv
```

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.