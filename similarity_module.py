import numpy as np
from scipy.stats import rankdata

def get_common_user_matrix(user_preference,user1_id,user2_id):
    movie_set = list(set(user_preference['movie_title'].values()))
    movie_dict =  {movie_set[x]:x for x in range(len(movie_set))}
    
    
    user_movie_rating = np.array([0,0,0])
 
    user_id_index = 0
    for i in  [user1_id,user2_id]:
        index = []
        for key,value in user_preference['user_id'].items():
            if value  == i:
                index.append(key)

        user_id =   [user_preference['user_id'].get(x)  for x in index]
        user_id = [user_id_index for x in user_id]
        movie_title = [user_preference['movie_title'].get(x)  for x in index]
        movie_id = [movie_dict[film] for film in movie_title]
        rating_ =   [user_preference['rating'].get(x)  for x in index]
        stack1 = np.stack([user_id,movie_id,rating_],axis = 1)
        user_movie_rating  = np.vstack([user_movie_rating,stack1])
        user_id_index += 1
    user_movie_rating = user_movie_rating[1:]
    
    
    n_items = max(np.unique(user_movie_rating[:,1]))
    n_users = len(np.unique(user_movie_rating[:,0]))
    data_matrix =  np.zeros((n_users,n_items))
    
    
    for line in user_movie_rating:
        data_matrix[line[0] -1,line[1] - 1 ] =  line[2]
        
    index_commonly_rated  = []
    for i in range(len(data_matrix[0])):
        d = data_matrix[:,i]
        if 0 in d:
            pass#arr.append(n)
        else:
            index_commonly_rated.append(i)
            
            
    common_matrix = data_matrix[:,index_commonly_rated]
    return common_matrix
    

def get_common_movie_matrix(user_preference,movie1_id,movie2_id):
    movie_set = list(set(user_preference['movie_title'].values()))
    movie_dict =  {movie_set[x]:x for x in range(len(movie_set))}

    user_movie_rating = np.array([0,0,0])

    movie_id_index = 0
    for movie in  [movie1_id,movie2_id]:
        index = []
        for key,value in user_preference['movie_title'].items():
            if value  == movie:
                index.append(key)

        user_id =   [user_preference['user_id'].get(x)  for x in index]

        movie_title = [user_preference['movie_title'].get(x)  for x in index]
        movie_id = [movie_dict[film] for film in movie_title]
        movie_id = [movie_id_index for x in movie_id]

        rating_ =   [user_preference['rating'].get(x)  for x in index]
        stack1 = np.stack([user_id,movie_id,rating_],axis = 1)
        user_movie_rating  = np.vstack([user_movie_rating,stack1])
        movie_id_index += 1
    user_movie_rating = user_movie_rating[1:]


    n_items = len(np.unique(user_movie_rating[:,1]))
    n_users = max(np.unique(user_movie_rating[:,0]))
    data_matrix =  np.zeros((n_users,n_items))


    for line in user_movie_rating:
        data_matrix[line[0] -1,line[1] - 1 ] =  line[2]

    data_matrix = data_matrix.T
    index_commonly_rated  = []
    for i in range(len(data_matrix[0])):
        d = data_matrix[:,i]
        if 0 in d:
            pass#arr.append(n)
        else:
            index_commonly_rated.append(i)


    common_matrix = data_matrix[:,index_commonly_rated]
    return common_matrix

def cosine_similarity(user_preference,id1,id2,recomendation_type = 'users' ):
    """
    cosine similarity is  sum(user1 * user2) / (square_root(sum(user1**2))  * square_root(sum(user2**2)))
    
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
    
    cosine_similarity_metric = sum_a_b/(sqrt_sum_a2 * sqrt_sum_b2)
    
    return cosine_similarity_metric

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
    
    squared_euclidean_similarity_metric = sum([(common_matrix[:,n][0] - common_matrix[:,n][1])**2   for n in range(len(common_matrix[0]))])
    return squared_euclidean_similarity_metric



def minkowski_distance_similarity(user_preference,id1,id2,n = 3,recomendation_type = 'users' ):
    """
    squared_euclidean_similarity is  ∑(user1[i] - user2[i])**2
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'

    minkowski_similarity_metric =(sum([(common_matrix[:,index][0] - common_matrix[:,index][1])**n    for index in range(len(common_matrix[0]))]))**(1/n)

    return minkowski_similarity_metric


def spearman_correlation_similarity(user_preference,id1,id2,recomendation_type = 'users' ):
    """
    squared_euclidean_similarity is  ∑(user1[i] - user2[i])**2
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'

    rank_data  = np.array([rankdata(common_matrix[index],'max')    for index in range(len(common_matrix))])
    array_len =  len(rank_data[0])
    spearman_correlation_metric = 1- (6* sum([(rank_data[:,n][0] - rank_data[:,n][1])**2    for n in range(array_len)])/(array_len*((array_len**2) - 1)))
    return spearman_correlation_metric

def chebyshev_similarity(user_preference,id1,id2,recomendation_type = 'users' ):
    """
    squared_euclidean_similarity is  ∑(user1[i] - user2[i])**2
    
    """
    if recomendation_type == 'users':
        common_matrix = get_common_user_matrix(user_preference,id1,id2)
    elif recomendation_type == 'movies':
        common_matrix = get_common_movie_matrix(user_preference,id1,id2)
    else:
        return 'Please enter a user id or movie id'

    chebyshev_similarity_metric = max([(common_matrix[:,n][0] - common_matrix[:,n][1])    for n in range(len(common_matrix[0]))])
    return chebyshev_similarity_metric