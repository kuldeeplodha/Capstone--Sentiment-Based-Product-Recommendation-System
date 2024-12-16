# Capstone - Sentiment Based Product Recommendation System


## Table of Contents
* [Problem Statement](#problem-statement)
* [Business goal](#business-goal)
* [Dataset](#dataset)
* [Expected tasks](#expected-tasks)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)

## Sentiment Based Product Recommendation System

### Problem Statement  

The e-commerce industry has seen tremendous growth, with businesses like Amazon, Flipkart, and Myntra dominating the market. To remain competitive, companies must provide superior customer experiences, including personalized product recommendations. Ebuss, a growing e-commerce company, is seeking to enhance its recommendation system by leveraging customer reviews and ratings. The aim is to use sentiment analysis to improve the accuracy and relevance of recommendations, thereby increasing customer satisfaction and sales.

---

### Business Goal 

The primary objective is to create a sentiment-based product recommendation system that improves the relevance of recommendations based on customer reviews and ratings. By implementing this system, Ebuss aims to:  
- Enhance customer satisfaction with tailored recommendations.  
- Compete effectively with established e-commerce platforms.  
- Drive sales and customer engagement by leveraging sentiment analysis insights.  

---

### Dataset  

The project uses a dataset consisting of **30,000 product reviews** across over **200 products** provided by **20,000 users**. The dataset includes customer reviews, ratings, and usernames. Key features include:  
- **reviews_username**: Identifies the user.  
- **Product reviews and ratings**: Provide the text and numerical rating data required for sentiment analysis and recommendations.  

The dataset has been sourced and preprocessed as a subset of a Kaggle competition, and detailed attribute descriptions are provided for reference.

---

### Expected Tasks  

The project is divided into the following tasks:  

1. **Data Sourcing and Sentiment Analysis**  
   - Perform exploratory data analysis (EDA).  
   - Clean and preprocess the text data (e.g., handle missing values, replace punctuations).  
   - Extract features using methods like TF-IDF, bag-of-words, or word embeddings.  
   - Train at least three machine learning models (e.g., Logistic Regression, Random Forest, XGBoost, Naive Bayes) for sentiment classification.  
   - Analyze model performance and select the best model.  

2. **Building a Recommendation System**  
   - Implement both user-based and item-based recommendation systems.  
   - Evaluate and select the best-suited system.  
   - Recommend 20 products for each user based on their interactions and ratings.  

3. **Improving Recommendations with Sentiment Analysis**  
   - Integrate the sentiment analysis model with the recommendation system.  
   - Filter the 20 recommended products to identify the top 5 products based on sentiment scores.  

4. **Deployment**  
   - Deploy the project using Flask to create a user-friendly web interface.  
   - Host the application on Heroku to make it accessible online.  
   - Implement a feature to input a username and display the top 5 recommendations for that user.  

---

### Technologies Used  

- **Python**: Core programming language for implementing the solution.  
- **NumPy**: Version 1.21.0
- **Pandas**: Version 1.3.4
- **Matplotlib**: Version 3.4.3
- **Seaborn**: Version 0.11.2
- **Scikit-learn**: Version 0.24.2
- **re**: Version 2.2.1
- **imbalanced-learn**: Version 0.8.0
- **nltk**: Version 3.6.7
- **TF-IDF/CountVectorizer**: For text feature extraction.  
- **Flask**: For deploying the ML model as a web application.  
- **Heroku**: For cloud-based hosting of the application.  

---

### Acknowledgements  
We would like to thank:  
- **Kaggle**: For providing the original dataset, which formed the basis of this project.  
- **Google Colab**: For offering a free platform to code and train models seamlessly.  
- The developers and maintainers of the libraries and tools used in this project, including **scikit-learn**, **Flask**, and **Heroku**.  



## Contact
Created by [Kuldeep Lodha](https://github.com/kuldeeplodha) - feel free to contact me!
