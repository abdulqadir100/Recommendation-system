import pandas as pd 
import numpy as np


def get_user_preference():
    # reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

    # reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

    # reading items file:
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
    'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
    encoding='latin-1')

    id_ =  items['movie id'].to_list()
    movie_name  = items['movie title'].to_list()

    movie_id = {x:y for x,y in zip(id_ , movie_name)}

    ratings['movie_title'] = ratings.movie_id.replace(movie_id)

    user_preference =  ratings[['user_id','movie_title','rating']].to_dict()
    
    return user_preference