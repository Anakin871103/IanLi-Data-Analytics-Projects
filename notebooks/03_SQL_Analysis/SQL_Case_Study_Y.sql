-- SQL_Case_Study_Y.sql

-- SQL Case Study Y: E-commerce Customer Behavior Analysis

-- This SQL script is designed to analyze customer behavior in an e-commerce setting.
-- It includes queries to extract insights from the customer transactions and interactions.

-- 1. Retrieve total sales per customer
SELECT 
    customer_id, 
    SUM(total_amount) AS total_sales
FROM 
    transactions
GROUP BY 
    customer_id
ORDER BY 
    total_sales DESC;

-- 2. Get the number of purchases per product
SELECT 
    product_id, 
    COUNT(*) AS purchase_count
FROM 
    transactions
GROUP BY 
    product_id
ORDER BY 
    purchase_count DESC;

-- 3. Find the average order value
SELECT 
    AVG(total_amount) AS average_order_value
FROM 
    transactions;

-- 4. Identify the top 10 customers by total sales
SELECT 
    customer_id, 
    SUM(total_amount) AS total_sales
FROM 
    transactions
GROUP BY 
    customer_id
ORDER BY 
    total_sales DESC
LIMIT 10;

-- 5. Analyze sales trends over time (monthly)
SELECT 
    DATE_TRUNC('month', transaction_date) AS month, 
    SUM(total_amount) AS monthly_sales
FROM 
    transactions
GROUP BY 
    month
ORDER BY 
    month;

-- 6. Get customer demographics
SELECT 
    customer_id, 
    age, 
    gender, 
    location
FROM 
    customers;

-- 7. Join customer data with transaction data for detailed analysis
SELECT 
    c.customer_id, 
    c.age, 
    c.gender, 
    SUM(t.total_amount) AS total_spent
FROM 
    customers c
JOIN 
    transactions t ON c.customer_id = t.customer_id
GROUP BY 
    c.customer_id, c.age, c.gender
ORDER BY 
    total_spent DESC;