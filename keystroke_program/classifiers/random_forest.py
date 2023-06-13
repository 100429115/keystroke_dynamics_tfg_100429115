from keystroke_program.classifiers.common_methods import data_label_extraction, data_feature_selection, performance_rates

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class RandomForest:

    def __init__(self, dataset, size=0.2, features="no_feature_selection", criterion=0,
                 depth=1, max_characteristics_number="1", estimators=1):

        self.classifier_name = "Random Forest"
        self.dataset = dataset

        # params
        if size < 0.0 or size > 1.0:
            self.size = 0.5
        else:
            self.size = size

        self.features = features
        self.criterion = criterion
        self.depth = depth
        self.max_characteristics_number = max_characteristics_number
        self.estimators = estimators

    def classification(self):
        # Random Forest classification process using the dataset requested
        data_label = data_label_extraction(self.dataset)

        if data_label == "Error":
            raise ValueError("Classification could not be done")

        data = data_label[0]
        labels = data_label[1]

        data = data_feature_selection(self.classifier_name, data, labels, self.dataset.value,
                                      self.features)

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.size, stratify=labels)

        random_forest = RandomForestClassifier(criterion=self.criterion,
                                               max_features=self.max_characteristics_number,
                                               max_depth=self.depth,
                                               n_estimators=self.estimators)

        random_forest.fit(x_train, y_train)
        y_pred = random_forest.predict(x_test)

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
    def criterion_getter(self):
        return self.criterion

    @criterion_getter.setter
    def criterion_getter(self, new_criterion):
        if new_criterion not in [0, 1]:
            raise ValueError("Sorry, your value does not accomplish the criteria")
        self.criterion = new_criterion

    @property
    def depth_getter(self):
        return self.depth

    @depth_getter.setter
    def depth_getter(self, new_depth):
        if type(new_depth) == int and new_depth > 0:
            self.depth = new_depth
        elif new_depth == str and new_depth.isdigit():
            self.depth = int(new_depth)

        raise ValueError("Sorry, your value does not accomplish the criteria")

    @property
    def max_characteristics_number_getter(self):
        return self.max_characteristics_number

    @max_characteristics_number_getter.setter
    def max_characteristics_number_getter(self, new_max_characteristics_number):
        if type(new_max_characteristics_number) == int and new_max_characteristics_number > 0:
            self.max_characteristics_number = str(new_max_characteristics_number)

        elif type(new_max_characteristics_number) == str:
            if new_max_characteristics_number.isdigit():
                self.max_characteristics_number = new_max_characteristics_number

            elif new_max_characteristics_number in ["sqrt", "log2"]:
                self.max_characteristics_number = new_max_characteristics_number
            else:
                raise ValueError("Sorry, your value does not accomplish the criteria")

        raise ValueError("Sorry, your value does not accomplish the criteria")

    @property
    def estimators_getter(self):
        return self.estimators

    @estimators_getter.setter
    def estimators_getter(self, new_estimators):
        if type(new_estimators) == int and new_estimators > 0:
            self.estimators = new_estimators
        elif new_estimators == str and new_estimators.isdigit():
            self.estimators = int(new_estimators)

        raise ValueError("Sorry, your value does not accomplish the criteria")
