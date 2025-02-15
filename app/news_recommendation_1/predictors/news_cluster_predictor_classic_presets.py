from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class NewsClusterClassicPredictorPresets:

    def __init__(self):
        pass

    def get_presets(self) -> dict:
        return {
            # "KNeighbors": {
            #     "estimator": KNeighborsClassifier(), # FALTA CONSIDERAR DESBALANCEAMENTO!
            #     "params": {
            #         'algorithm': ['ball_tree'],
            #         'n_neighbors': [35],
            #         'leaf_size': [1],
            #         'p': [2],
            #         'weights': ['uniform'],
            #         'metric': ['minkowski']
            #     }
            # },
            "SupportVectorMachine": {
                "estimator": LinearSVC(class_weight='balanced'),
                "params": {
                    'penalty': ['l2'],
                    'C': [1],
                }
            },
            "DecisionTree": {
                "estimator": DecisionTreeClassifier(class_weight='balanced'),
                "params": {
                    'criterion': ("gini",),
                    'max_depth': (5,),
                    'splitter': ('best',)
                }
            },
            "RandomForest": {
                "estimator": RandomForestClassifier(class_weight='balanced'),
                "params": {
                    'n_estimators': [50],
                    'max_depth': [10]
                }
            },
            "XGBClassifier": {
                "estimator": XGBClassifier(),
                "params": {
                    'learning_rate': [0.1],
                    'max_depth': [5],
                    'n_estimators': [110],
                }
            },
        }

    # def get_presets(self) -> dict:
    #     return {
    #         "KNeighborsClassifier": {
    #             "estimator": KNeighborsClassifier(),
    #             "params": {
    #                 'algorithm': ('ball_tree', 'kd_tree', 'brute'),
    #                 'n_neighbors': (3, 5, 7, 9),
    #                 'leaf_size': (20, 40, 1),
    #                 'p': (1, 2),
    #                 'weights': ('uniform', 'distance'),
    #                 'metric': ('minkowski', 'chebyshev')
    #             }
    #
    #         },
    #         "SVC": {
    #             "estimator": SVC(),
    #             "params": {
    #                 'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
    #                 'C': (1, 10, 100)
    #             }
    #         },
    #         "DecisionTreeClassifier": {
    #             "estimator": DecisionTreeClassifier(),
    #             "params": {
    #                 'criterion': ("gini", "entropy", "log_loss"),
    #                 'max_depth': (None, 5, 10)
    #             }
    #         },
    #         "RandomForestClassifier": {
    #             "estimator": RandomForestClassifier(),
    #             "params": {
    #                 'n_estimators': (10, 50, 100),
    #                 'max_depth': (None, 5, 10)
    #             }
    #         },
    #         "XGBClassifier": {
    #             "estimator": XGBClassifier(),
    #             "params": {
    #                 'learning_rate': [0.01, 0.1, 0.2],
    #                 'max_depth': [3, 5, 7],
    #                 'n_estimators': [100, 200, 300],
    #             }
    #         },
    #     }