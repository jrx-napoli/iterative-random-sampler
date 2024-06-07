from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def random_forest_classifier():
    model = RandomForestClassifier().fit
    return RandomForestClassifier()


def random_forest_regressor():
    return RandomForestRegressor()
