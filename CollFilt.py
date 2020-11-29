import argparse
import json
import numpy as np

from SimScore import pearson_score

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find projects who are similar to the input project')
    parser.add_argument('--project', dest='project', required=True,
            help='Input project')
    return parser
 
def find_similar_projects(dataset, project, num_projects):
    if project not in dataset:
        raise TypeError('Cannot find ' + project + ' in the dataset')

    scores = np.array([[x, pearson_score(dataset, project, x)] for x in dataset if x != project])

    scores_sorted = np.argsort(scores[:, 1])[::-1]

    top_projects = scores_sorted[:num_projects] 

    return scores[top_projects] 

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    project = args.project

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    print('\nprojects similar to ' + project + ':\n')
    similar_projects = find_similar_projects(data, project, 3) 
    print('project\t\t\tSimilarity score')
    print('-'*41)
    for item in similar_projects:
        print(item[0], '\t\t', round(float(item[1]), 2))