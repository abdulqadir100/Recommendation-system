from load_dataset_module import get_user_preference
from similarity_module import (cosine_similarity,squared_euclidean_similarity,minkowski_distance_similarity,
                               spearman_correlation_similarity,chebyshev_similarity,hamming_distance_similarity)

# loading the dataset
user_preference =  get_user_preference()
# This function takes in input to detemine movie similarity or user similarity
def user_input(similarity_metric = 'cosine'):
    similarity_type = input("enter similarity type as either 'users' or 'movies' ")
    if similarity_type == 'users':
        user_similarity_type()
    elif similarity_type == 'movies':
        movie_similarity_type()
    else:
        print("enter similarity type as either 'users' or 'movies' ")
        return user_input()
    status_of_exit()
    
# This function takes in two user_id input from the user if the function input of "user_input"   is users
def user_similarity_type():
    try:
        user1 =  int(input('Enter the ID of the first user: ' ))
        user2 =  int(input('Enter the ID of the second user: ' ))
        users_similarity_calculated =  similarity_metric_user(user1,user2)
        print('The similarity between user: ',user1,' and user: ',user2, ' is : ',users_similarity_calculated)
    except ValueError:
        print('ENTER THE USER_ID AS A NUMBER')
        return user_similarity_type()

    
# This function returns the user similarity based on the selected metric
def similarity_metric_user(user1,user2):
    param = {'user_preference':user_preference, "id1":user1, "id2":user2}
    print('similarity_metric_type : cosine, sq_euclidean,minkowski,spearman,chebyshev,hamming')
    similarity_metric_type  = input('Enter the ID of the similarity_metric_type: ' )
    metrics = {'cosine': cosine_similarity(**param),'sq_euclidean':squared_euclidean_similarity(**param),'minkowski':minkowski_distance_similarity(**param),
              'spearman': spearman_correlation_similarity(**param),'chebyshev': chebyshev_similarity(**param),'hamming': hamming_distance_similarity(**param)}
    try:
        similarity = metrics[similarity_metric_type]
        return similarity
    except KeyError:
        return similarity_metric_user(user1,user2)

    

# This function takes in two movie input from the user if the function input of "user_input"   is movies
def movie_similarity_type():
    try:
        movie1 =  input('Enter the Title of the first movie : ' )
        movie2 =  input('Enter the Title of the second movie : ' )
        movie_similarity_calculated =  similarity_metric_movie(movie1,movie2)
        print('The similarity between the movie : ',movie1,' and the movie: ',movie2, ' is : ',movie_similarity_calculated)
    except ValueError:
        print('ENTER THE MOVIE_TITLE AS A STRING')
        return movie_similarity_type()
    
    
# This function returns the movie similarity based on the selected metric
def similarity_metric_movie(movie1,movie2):
    param = {'user_preference':user_preference, "id1":movie1, "id2":movie2,'recomendation_type':'movies'}
    print('similarity_metric_type : cosine, sq_euclidean,minkowski,spearman,chebyshev,hamming')
    similarity_metric_type  = input('Enter the ID of the similarity_metric_type: ' )
    metrics = {'cosine': cosine_similarity(**param),'sq_euclidean':squared_euclidean_similarity(**param),'minkowski':minkowski_distance_similarity(**param),
              'spearman': spearman_correlation_similarity(**param),'chebyshev': chebyshev_similarity(**param),'hamming': hamming_distance_similarity(**param)}
    try:
        similarity = metrics[similarity_metric_type]
        return similarity
    except KeyError:
        return similarity_metric_movie(movie1,movie2)

    

# This function exits the recursion if input entered is 0
def status_of_exit():
    exit_status = int(input('Enter the exit status as 1 to continue or 0 to stop') )
    if exit_status == 1:
        return user_input()
    elif exit_status == 0:
        print('Thanks for using this recommender engine ; have a nice day')
    else:
        print('Enter the exit status as 1 to continue or 0 to stop')
        return status_of_exit()
