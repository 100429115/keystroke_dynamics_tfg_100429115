from keystroke_program.classifiers.common_methods import data_label_extraction, data_feature_selection, performance_rates

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNN:

    def __init__(self, dataset, size=0.2, features="no_feature_selection", neighbors=1,
                 metric="manhattan", weight="uniform"):

        self.dataset = dataset
        self.classifier_name = "KNN"

        # params
        if size < 0.0 or size > 1.0:
            self.size = 0.5
        else:
            self.size = size
        self.features = features
        self.neighbors = neighbors
        self.metric = metric
        self.weight = weight

        self.metrics = ["l1", "braycurtis", "canberra", "chebyshev", "cityblock", "correlation",
                        "cosine", "euclidean", "jensenshannon", "mahalanobis", "minkowski",
                        "seuclidean", "sqeuclidean"]

    def classification(self):
        # KNN classification process using the dataset requested
        data_label = data_label_extraction(self.dataset)

        if data_label == "Error":
            raise ValueError("Classification could not be done")

        data = data_label[0]
        labels = data_label[1]

        data = data_feature_selection(self.classifier_name, data, labels, self.dataset.value,
                                      self.features, self.neighbors, self.metric, self.weight)

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.size, stratify=labels)

        knn = KNeighborsClassifier(n_neighbors=self.neighbors, metric=self.metric, weights=self.weight)

        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        # quality of results
        rates = performance_rates(y_test, y_pred)
        f1 = rates[0]
        accuracy = rates[1]
        false_positive = rates[2]
        false_negative = rates[3]

        result = "\n\nf1: " + str(f1) + "\n\naccuracy: " + str(accuracy) + "\n\nfalse positive rate: " \
                 + str(false_positive) + "\n\nfalse negative rate: " + str(false_negative)

        return result

    # getters and setters

    @property
    def size_getter(self):
        return self.size

    @size_getter.setter
    def size_getter(self, new_size):
        if (type(new_size) != float) or not (0.0 < new_size < 1.0):
            raise ValueError("Sorry, your value does not accomplish the criteria")
        self.size = new_size

    @property
    def features_getter(self):
        return self.features

    @features_getter.setter
    def features_getter(self, new_features):
        if type(new_features) != str:
            raise ValueError("Sorry, your value does not accomplish the criteria")
        self.features = new_features

    @property
    def neighbors_getter(self):
        return self.neighbors

    @neighbors_getter.setter
    def neighbors_getter(self, new_neighbors):
        if type(new_neighbors) == str and new_neighbors.isdigit():
            self.neighbors = int(new_neighbors)
        elif type(new_neighbors) == int:
            self.neighbors = new_neighbors
        raise ValueError("Sorry, your value does not accomplish the criteria")

    @property
    def metric_getter(self):
        return self.metric

    @metric_getter.setter
    def metric_getter(self, new_metric):
        if new_metric not in self.metrics:
            raise ValueError("Sorry, your value does not accomplish the criteria")
        self.metric = new_metric

    @property
    def weight_getter(self):
        return self.weight

    @weight_getter.setter
    def weight_getter(self, new_weight):
        if new_weight not in ["uniform", "distance"]:
            raise ValueError("Sorry, your value does not accomplish the criteria")
        self.weight = new_weight
