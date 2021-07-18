import numpy as np
from scipy.stats import rankdata

def get_common_user_matrix(user_preference,user1_id,user2_id):
    """
    This returns a matrix of movies related to two users
    """
    
    # Generates a set of unique movie names
    movie_set = list(set(user_preference['movie_title'].values()))
    # Assigns an index to the movies
    movie_dict =  {movie_set[x]:x for x in range(len(movie_set))}
    
    # Creates a dummy array for appending a user_id , movie_id and rating
    user_movie_rating = np.array([0,0,0])
    # reindexes the user_id from 0 for easier array slicing operation
    user_id_index = 0
    # Generates a list of the different index at which the two unique users appears in the dataset
    for i in  [user1_id,user2_id]:
        index = []
        for key,value in user_preference['user_id'].items():
            if value  == i:
                index.append(key)
        # Generates the user_id for each user
        user_id =   [user_preference['user_id'].get(x)  for x in index]
        # Reindexing happens here
        user_id = [user_id_index for x in user_id]
        # Generates the equivalent movie title for a user
        movie_title = [user_preference['movie_title'].get(x)  for x in index]
        # Changing the movie title to movie_id for easier array slicing operations
        movie_id = [movie_dict[film] for film in movie_title]
        # Generates the equivalent movie rating for a user
        rating_ =   [user_preference['rating'].get(x)  for x in index]
        # Merges the user_id , movie_id and rating for  a user
        stack1 = np.stack([user_id,movie_id,rating_],axis = 1)
        # merges the user_id , movie_id and rating of each user together
        user_movie_rating  = np.vstack([user_movie_rating,stack1])
        user_id_index += 1
    # Dropping the dummy array
    user_movie_rating = user_movie_rating[1:]
    
    # Getting the number of movies watched by the users
    n_items = max(np.unique(user_movie_rating[:,1]))
    # Getting the number to users who watched a movie
    n_users = len(np.unique(user_movie_rating[:,0]))
    
    # Creating a data matrix from the number movies watched and the number of users
    data_matrix =  np.zeros((n_users,n_items))
    
    # places a rating at the index of the movie and user in the data matrix
    for line in user_movie_rating:
        data_matrix[line[0] -1,line[1] - 1 ] =  line[2]
    
    # Generates the index of movies common to the two users
    index_commonly_rated  = []
    for i in range(len(data_matrix[0])):
        d = data_matrix[:,i]
        if 0 in d:
            pass
        else:
            index_commonly_rated.append(i)
            
    # selecting the common movies in the data matrix
    common_matrix = data_matrix[:,index_commonly_rated]
    # returns the common movies matrix for the two users
    return common_matrix
    

def get_common_movie_matrix(user_preference,movie1_id,movie2_id):
    """
    This returns a matrix of user related to two movies
    """
    # Generates a set of unique movie names
    movie_set = list(set(user_preference['movie_title'].values()))
    # Assigns an index to the movies
    movie_dict =  {movie_set[x]:x for x in range(len(movie_set))}

    # Creates a dummy array for appending a user_id , movie_id and rating
    user_movie_rating = np.array([0,0,0])

    # reindexes the movie_id from 0 for easier array slicing operation
    movie_id_index = 0
    # Generates a list of the different index at which the two unique movies appears in the dataset
    for movie in  [movie1_id,movie2_id]:
        index = []
        for key,value in user_preference['movie_title'].items():
            if value  == movie:
                index.append(key)

        # Generates the user_id for each movie
        user_id =   [user_preference['user_id'].get(x)  for x in index]
        # Generates the equivalent movie title 
        movie_title = [user_preference['movie_title'].get(x)  for x in index]
        # Changing the movie title to movie_id for easier array slicing operations
        movie_id = [movie_dict[film] for film in movie_title]
        # Reindexing happens here
        movie_id = [movie_id_index for x in movie_id]
        # Generates the equivalent user rating for a movie
        rating_ =   [user_preference['rating'].get(x)  for x in index]
        # Merges the user_id , movie_id and rating for  a movie
        stack1 = np.stack([user_id,movie_id,rating_],axis = 1)
        # merges the user_id , movie_id and rating of each movie together
        user_movie_rating  = np.vstack([user_movie_rating,stack1])
        movie_id_index += 1
        
    # Dropping the dummy array
    user_movie_rating = user_movie_rating[1:]

    # Getting the number of movies watched by the users
    n_items = len(np.unique(user_movie_rating[:,1]))
    # Getting the number to users who watched a movie
    n_users = max(np.unique(user_movie_rating[:,0]))
    # Creating a data matrix from the number movies watched and the number of users
    data_matrix =  np.zeros((n_users,n_items))

     # places a rating at the index of the movie and user in the data matrix
    for line in user_movie_rating:
        data_matrix[line[0] -1,line[1] - 1 ] =  line[2]
    # Generates the index of users common to the two movies
    data_matrix = data_matrix.T
    index_commonly_rated  = []
    for i in range(len(data_matrix[0])):
        d = data_matrix[:,i]
        if 0 in d:
            pass#arr.append(n)
        else:
            index_commonly_rated.append(i)

    # selecting the common users for the two movies in the data matrix
    common_matrix = data_matrix[:,index_commonly_rated]
    # returns the common user matrix for the two movies
    return common_matrix

# This fuction calculate the cosine_similarity
def cosine_similarity(user_preference,id1,id2,recomendation_type = 'users' ):
    """
    Cosine similarity measures the similarity between two vectors of an inner product space .It is the  sum(user1 * user2) / (square_root(sum(user1**2))  * square_root(sum(user2**2)))
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'
   
    sum_a_b  = sum([common_matrix[:,i][0]*common_matrix[:,i][1] for i in range(len(common_matrix[0]))])
    sqrt_sum_a2  = np.sqrt(sum([common_matrix[:,i][0]*common_matrix[:,i][0] for i in range(len(common_matrix[0]))]))
    sqrt_sum_b2  = np.sqrt(sum([common_matrix[:,i][1]*common_matrix[:,i][1] for i in range(len(common_matrix[0]))]))
    
    try:
        cosine_similarity_metric = sum_a_b/(sqrt_sum_a2 * sqrt_sum_b2)
    except ZeroDivisionError:
        cosine_similarity_metric = -1000
    return cosine_similarity_metric

# This fuction calculate the squared_euclidean_similarity
def squared_euclidean_similarity(user_preference,id1,id2,recomendation_type = 'users' ):
    """
    squared_euclidean_similarity is  ∑(user1[i] - user2[i])**2
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'
    
    try:
        squared_euclidean_similarity_metric = sum([(common_matrix[:,n][0] - common_matrix[:,n][1])**2   for n in range(len(common_matrix[0]))])
    except ZeroDivisionError:
        squared_euclidean_similarity_metric = -1000
    return squared_euclidean_similarity_metric


# This fuction calculates the minkowski_distance_similarity
def minkowski_distance_similarity(user_preference,id1,id2,n = 3,recomendation_type = 'users' ):
    """
    The Minkowski distance or Minkowski metric is a metric in a normed vector space which can be considered as a generalization of both the Euclidean distance and the Manhattan distance
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'
    try:
        minkowski_similarity_metric =(sum([(common_matrix[:,index][0] - common_matrix[:,index][1])**n    for index in range(len(common_matrix[0]))]))**(1/n)
    except ZeroDivisionError:
        minkowski_similarity_metric = -1000
    return minkowski_similarity_metric


# This fuction calculates the spearman_correlation_similarity
def spearman_correlation_similarity(user_preference,id1,id2,recomendation_type = 'users' ):
    """
    spearman_correlation assesses how well the relationship between two variables can be described using a monotonic function. 
    the Spearman correlation between two variables will be high when observations have a similar (or identical for a correlation of 1) rank (i.e. relative position label of the observations within the variable: 1st, 2nd, 3rd, etc.) between the two variables, and low when observations have a dissimilar (or fully opposed for a correlation of −1) rank between the two variables
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'

    rank_data  = np.array([rankdata(common_matrix[index],'max')    for index in range(len(common_matrix))])
    array_len =  len(rank_data[0])
    try :
        spearman_correlation_metric = 1- (6* sum([(rank_data[:,n][0] - rank_data[:,n][1])**2    for n in range(array_len)])/(array_len*((array_len**2) - 1)))
    except ZeroDivisionError:
        spearman_correlation_metric = -1000
    return spearman_correlation_metric

# This fuction calculates the chebyshev_similarity
def chebyshev_similarity(user_preference,id1,id2,recomendation_type = 'users' ):
    """
    Chebyshev distance, maximum metric, or L∞ metric is a metric defined on a vector space where the distance between two vectors is the greatest of their differences along any coordinate dimension
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'
    try:
        chebyshev_similarity_metric = max([(common_matrix[:,n][0] - common_matrix[:,n][1])    for n in range(len(common_matrix[0]))])
    except ZeroDivisionError:
        chebyshev_similarity_metric = -1000
    return chebyshev_similarity_metric


# This fuction calculates the hamming_distance_similarity
def hamming_distance_similarity(user_preference,id1,id2,recomendation_type = 'users' ):
    """
    Hamming Distance measures the similarity between two users or movies based on the number of ratings that are not equivalent
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'
    try:
        hamming_similarity_metric = len(([1  for n in range(len(common_matrix[0])) if common_matrix[:,n][0] != common_matrix[:,n][1]]))
    except ZeroDivisionError:
        hamming_similarity_metric = -1000
    return hamming_similarity_metric