---
layout: post
title: Building a Movie Recommender System with Collaborative Filtering
subtitle: Using Apache Spark (Pyspark)
cover-img: "https://github.com/user-attachments/assets/6acb1830-2047-45ea-a035-96a4a27e0efd"
thumbnail-img: "https://github.com/user-attachments/assets/fdf52c45-4b8c-42b0-8076-0e811a400f8b"
share-img: "https://github.com/user-attachments/assets/6acb1830-2047-45ea-a035-96a4a27e0efd"
gh-repo: nazpe/Thesis
gh-badge: [star, follow]
tags: [Project]
comments: true
mathjax: true
author: Nuno Pedrosa
---

## Unveiling Movie Preferences: A Deep Dive into Collaborative Filtering with Spark

Recommender systems are a fundamental component of modern digital platforms, helping users discover relevant content, products, or services by filtering vast amounts of information. From suggesting movies on Netflix to recommending products on Amazon, these systems aim to personalize the user experience and increase engagement. There are three primary approaches to building recommender systems: content-based filtering, collaborative filtering, and latent factor models. Each approach has its unique methodology, strengths, and limitations.

**Content-based filtering** recommends items to a user based on the characteristics of items they have previously liked. It relies heavily on item metadata — such as genre, category, or descriptive keywords — to find similar items. This approach is highly personalized and works well even with a small user base. However, it tends to narrow user choices by repeatedly recommending similar types of items, potentially limiting discovery.

**Collaborative filtering** focuses on user behavior rather than item attributes. It identifies users with similar preferences and suggests items that those users have liked, assuming that users who agreed in the past will likely agree again. This method can uncover surprising or unexpected recommendations by leveraging community behavior, but it struggles with new users or new items — a challenge known as the cold start problem.

**Latent factor models**, such as matrix factorization, use mathematical techniques to discover hidden patterns in user-item interactions. These models reduce the complex interaction matrix into lower-dimensional representations, capturing abstract factors like "taste" or "style" without needing explicit item features. While often more accurate and scalable, latent factor models are less interpretable and also suffer from cold start limitations.

In this post, I'll walk you through a project where I implemented **collaborative filtering** using Apache Spark to build a movie recommendation system based on the MovieLens dataset.

### The Data Behind the Magic

My project utilized the `ml-latest-small` dataset from MovieLens, a fantastic resource for exploring recommendation systems retrieved by the University of Minnesota. This dataset contains:

*   **100,836 ratings** and **3,683 tag applications** across **9,742 movies**.
*   Data from **610 users** who each rated at least 20 movies.
*   Key files: `ratings.csv` (user, movie, rating, timestamp), `movies.csv` (movie ID, title, genres).

The core idea is to leverage the `ratings.csv` file, which tells us how users have rated different movies. This becomes the foundation for understanding user preferences and movie similarities.

### Initial Data Preparation

First things first, I set up a Spark session – a crucial step for handling large datasets efficiently.

```python

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession \
    .builder \
    .master('local[*]') \
    .config("spark.driver.memory", "15g") \
    .appName("MovieLens CF") \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

Movies_df = spark.read.csv("ml-latest-small/movies.csv",header=True)
Ratings_df = spark.read.csv("ml-latest-small/ratings.csv",header=True)
```

![Extraction Pipeline](https://github.com/user-attachments/assets/c8205bff-e2e2-48ec-aafe-87b1efaebfe2){: .mx-auto.d-block :}

To understand the distribution of our data, here are some small samples.

### Training and Testing Split

To validate the performance of our recommendation system, the dataset was split into training (90%) and testing (10%) sets. This allows us to train the model on a majority of the data and then evaluate its predictions on unseen data.

```python
(Train, Test) = Ratings_df.randomSplit([0.9, 0.1], seed = 0)
```

An important check involved ensuring that all users from the test set were also present in the training set. While users were consistent, some movies in the test set were not in the training set; ratings for these "new" movies would have to be ignored during validation.

### Constructing the User-Item Matrix

The heart of many collaborative filtering algorithms is the user-item matrix, where rows represent users, columns represent movies, and cell values are ratings. I constructed this in two ways: one for calculating movie similarities (movie-based RDD) and one for predicting user ratings (user-based RDD).

This involved custom functions (`transformRating` and `RatingJunction`) to efficiently populate the RDDs.

```python
# Custom functions for RDD construction
def transformRating(Id_1, rating, Id_2, items):
    rating_list = [rating if ele == Id_1 else None for ele in items]
    return ([Id_2]+[rating_list])

def RatingJunction(a,b):
    n=0
    for ind in b:
        if ind != None:
            break
        n=n+1
    c=a
    c[n]=b[n]
    return c

# Example for User-based RDD
items_movies = Train.select('movieId').rdd.map(lambda data:data.movieId).collect()
items_movies = list(dict.fromkeys(items_movies))
Train_user_RDD = Train.rdd.map(lambda data:(data.movieId,data.rating,data.userId))
Train_user_RDD = Train_user_RDD.map((lambda data:transformRating(data[0],data[1],data[2],items_movies)))
Train_user_RDD = Train_user_RDD.map(lambda item: (item[0],item[1]))
Train_user_RDD = Train_user_RDD.reduceByKey(lambda data_1,data_2:RatingJunction(data_1,data_2))
```

The resulting RDDs look something like this, with `None` representing unrated movies/users:

```
+---+--------------------+
| _1|                  _2|
+---+--------------------+
|  1|[4.0, 4.0, 4.0, 5...|
|  2|[null, null, null...|
...
+---+--------------------+
```

### Calculating Movie Similarities

To recommend movies, we need to know how similar they are to each other. I used **Pearson correlation** by first normalizing the ratings (subtracting the mean rating for each movie) and then calculating the cosine similarity between the adjusted rating vectors. Ratings that didn't exist were treated as 0.

```python
def Pearson_step1(item):
    ratings=item[1]
    ratings_Ex = list(filter(None,ratings))
    mean=sum(ratings_Ex)/len(ratings_Ex)
    n=0
    for rat in ratings:
        if rat != None:
            ratings[n]=ratings[n]-mean
        else:
            ratings[n]=0.0
        n=n+1
    return (item[0], ratings)

Similarity_RDD = Train_movie_RDD.map(lambda item: Pearson_step1(item))
```

After obtaining the normalized ratings, I performed a cross-join to compare every movie with every other movie and then applied a `cosine_sim` function to compute the actual similarity.

```python
def cosine_sim(item):
    rating_1=item[1]
    rating_2=item[2]
    prod_list=[]
    for n in range(0,item_users_len):
        number=rating_1[n]*rating_2[n]
        prod_list.append(number)
    prod=sum(prod_list)
    square_1=sqrt(sum([ x**2 for x in rating_1 ]))
    square_2=sqrt(sum([ x**2 for x in rating_2 ]))
    prod2=square_1*square_2
    if prod2==0:
        prod2=0.000000000000000001 # Avoid division by zero
    similarity=prod/prod2
    return (item[0],similarity)

similarityRDD=JoinedRDD.map(lambda data: cosine_sim(data))
```

To optimize and focus on meaningful relationships, I filtered out similarities below a threshold of 0.3, considering them weak correlations. This significantly reduces computation time later on.

### Predicting Scores for Unrated Movies

With movie similarities in hand, the next step was to predict ratings for movies a user hasn't seen yet. For each unrated movie by a user:

1.  I identified the 10 most similar movies that the user **had** rated.
2.  The predicted score was then calculated as the weighted average of the user's ratings for these similar movies, where the weights are the similarity scores.

```python
def scores(item):
    user=item[0]
    ratings_change=item[1]
    ratings_non_change=ratings_change[:]
    
    for n in range(0,item_movies_len):
        if ratings_change[n]==None:
            i=items_movies[n]            
            i_dict = {}
            for item, value in Similarity_Dict.items():
                if (item[0] == i) and ratings_non_change[items_movies.index(item[1])]!=None:
                    i_dict[item] = (value, ratings_non_change[items_movies.index(item[1])])                 
            i_dict = sorted(i_dict.items(), key=lambda x:-x[1][0])[:10]           
            
            term1=0
            term2=0
            for item, value in i_dict:
                term1=term1+(value[0]*value[1])
                term2=term2+value[0]
            if term2==0:
                term2=0.0000000000000000001
            score=term1/term2
            ratings_change[n]=score
        else:
            ratings_change[n]=-1 # Mark as already rated
            
    return (user,ratings_change)

ScoresRDD=Train_user_RDD.map(lambda data: scores(data))
```

### Generating Recommendations

After predicting scores for all unrated movies for all users, I picked a few users to showcase the recommendations. For each selected user, I listed movies with a predicted rating of 4.5 or higher.

Here are some example recommendations for User 1:

*   Wolf of Wall Street, The (2013)
*   Interstellar (2014)
*   Whiplash (2014)
*   Dangerous Minds (1995)
*   ... and many more!

### Validating the Results: RMSE and Precision@10

To understand how well our model performs, I used two key metrics:

1.  **Root Mean Squared Error (RMSE):** This metric tells us the average magnitude of the errors between predicted and actual ratings. A lower RMSE indicates better accuracy.
2.  **Precision@10:** This measures, out of the top 10 recommended movies, how many were actually liked by the user (based on their real ratings in the test set, above a certain threshold, e.g., 3.5 stars).

My model achieved an **RMSE of 1.17**. This means, on average, our predicted ratings were off by about 1.17 stars from the user's actual ratings. While not perfect, it's a reasonable starting point for identifying movies a user might enjoy.

For Precision@10, the results varied by user. User 1 had an impressive 90% precision, meaning 9 out of their top 10 recommendations were genuinely liked. Other users showed varying degrees of success. The overall mean Precision@10 across the analyzed users was **0.67**.

```python
# RMSE calculation
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction") 
RMSE = evaluator.evaluate(Test_predict)
print(f"RMSE: {RMSE}")

# Precision@10 calculation
# ... (code as provided in your project) ...
print('Top 10 algorithm mean:')
print(Top10_mean)
```

The variation in Precision@10 highlights an important aspect: the performance of recommendation systems can differ across users, especially if some users have fewer ratings in the test set that meet the comparison criteria.

### Conclusion

This project demonstrates a fundamental collaborative filtering approach using Spark. While more advanced techniques like Alternating Least Squares (ALS) can often yield better results (and are built into Spark MLlib!), this direct RDD-based implementation provides a solid understanding of the underlying mechanics of user-item similarity and rating prediction. The results, with an RMSE of 1.17 and a mean Precision@10 of 0.67, show that even a basic collaborative filtering system can provide valuable movie recommendations.


