import argparse
import json
import numpy as np

from SimScore import pearson_score
from CollFilt import find_similar_projects

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find the employee recommendations for the given project')
    parser.add_argument('--project', dest='project', required=True,
            help='Input project')
    return parser
 
def get_recommendations(dataset, input_project):
    if input_project not in dataset:
        raise TypeError('Cannot find ' + input_project + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    for project in [x for x in dataset if x != input_project]:
        similarity_score = pearson_score(dataset, input_project, project)

        if similarity_score <= 0:
            continue
        
        filtered_list = [x for x in dataset[project] if x not in \
                dataset[input_project] or dataset[input_project][x] == 0]

        for item in filtered_list: 
            overall_scores.update({item: dataset[project][item] * similarity_score})
            similarity_scores.update({item: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    movie_scores = np.array([[score/similarity_scores[item], item] 
            for item, score in overall_scores.items()])

    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    movie_recommendations = [movie for _, movie in movie_scores]

    return movie_recommendations
 
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    project = args.project

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    print("\Employee recommendations for " + project + ":")
    movies = get_recommendations(data, project) 
    for i, movie in enumerate(movies):
        print(str(i+1) + '. ' + movie)