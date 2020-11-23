from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline   import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

X, y = samples_generator.make_classification(n_samples = 150,
        n_features = 25, n_classes = 3, n_informative = 6,
        n_redundant = 0, random_state = 7)

k_best_selector = SelectKBest(f_regression, k = 9)
classifier = ExtraTreesClassifier(n_estimators = 60, max_depth = 4)
processor_pipeline = Pipeline([('selector', k_best_selector), ('erf', classifier)])

processor_pipeline.set_params(selector__k = 7, erf__n_estimators = 30)
processor_pipeline.fit(X, y)
output = processor_pipeline.predict(X)
print("\npredicted output:\n", output)
print("\nScore:", processor_pipeline.score(X,y))

status = processor_pipeline.named_steps['selector'].get_support()

selected = [i for i, x in enumerate(status) if x]
print("\nIndices of selected features:",', '.join([str(x) for x in selected]))