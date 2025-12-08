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
