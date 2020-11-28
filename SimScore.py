import argparse
import json
import numpy as np

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--project1', dest='project1', required=True,
            help='First project')
    parser.add_argument('--project2', dest='project2', required=True,
            help='Second project')
    return parser



def pearson_score(dataset, project1, project2):
    if project1 not in dataset:
        raise TypeError('Cannot find ' + project1 + ' in the dataset')

    if project2 not in dataset:
        raise TypeError('Cannot find ' + project2 + ' in the dataset')

    common_emolyees = {}

    for item in dataset[project1]:
        if item in dataset[project2]:
            common_emolyees[item] = 1

    num_ratings = len(common_emolyees) 

    if num_ratings == 0:
        return 0

    project1_sum = np.sum([dataset[project1][item] for item in common_emolyees])
    project2_sum = np.sum([dataset[project2][item] for item in common_emolyees])

    project1_squared_sum = np.sum([np.square(dataset[project1][item]) for item in common_emolyees])
    project2_squared_sum = np.sum([np.square(dataset[project2][item]) for item in common_emolyees])

    sum_of_products = np.sum([dataset[project1][item] * dataset[project2][item] for item in common_emolyees])

    Sxy = sum_of_products - (project1_sum * project2_sum / num_ratings)
    Sxx = project1_squared_sum - np.square(project1_sum) / num_ratings
    Syy = project2_squared_sum - np.square(project2_sum) / num_ratings
    
    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    project1 = args.project1
    project2 = args.project2

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    print("\nPearson score:")
    print(pearson_score(data, project1, project2))