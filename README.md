# Netflix-Clone-DS5110-project
DS5110 - Final Project

Author: Liam Campbell

## Project Goal
The goal of this project is to build a Netflix-style movie recommendation system using the MovieLens 100k dataset.
This project evaluates how different recommendation approaches perform when generating personalized movie suggestions for users based on their historical viewing and rating behavior.

## Key Components of the System

* SQL Database: Fully structured database containing movies, users, ratings, and metadata

* Three Recommendation Models:

  * Collaborative Filtering (CF) using SVD

  * Content-Based Filtering (CBF) using TF-IDF + cosine similarity

  * Hybrid Model combining CF predictions with content similarity

* Flask Web Application: Interactive front-end styled after Netflix, displaying recommendations in horizontal rows

## Additional Insights:

* Trending Now – movies popular across many users

* Classics – highly-rated historical films

* Because You Watched – content-based similarity recommendations

## How To run:
* Clone the Repository
* Install Dependencies
* Instal required packages: -r requirements.txt
* Run the preprocessing script to build SQLite DB
* Train the models and generate the .pkl files
  * train_cf.py
  * train_content_based.py
  * train_hybrid.py
* Run Flask App
  * From project root: python/src/app/app.py

## Methodolgy Overview
* Collaborative Filtering (SVD)
  * Learns latent factors between users and movies
  * Predict missing ratings
* Content-Based Filtering
  * TF-IDF vectorization of genres
  * Cosine similarity to find similar movies
* Hybrid Approach
  * Weighted combination of Collaborative Filtering predictions and movie similarity
  * Improves diversity and cold-start issues
* Additional Insights
  * Trending (most rated movies)
  * Classics (Older, highly rated movies)
  * Because You Watched (similarity-based row)

 ## Results
 
 Collaborative Filtering RMSE: 0.93
 
 Qualitative Findings include
 * Hybrid model improves personalization
 * Content-based filtering performs well even with sparse data
 * Genre provides strong similarity signals
  
