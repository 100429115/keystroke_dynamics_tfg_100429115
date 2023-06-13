
from keystroke_program.classifiers.common_methods import data_label_extraction, data_feature_selection, performance_rates

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB


class NaiveBayes:

    def __init__(self, dataset, size=0.2, features="no_feature_selection", naive_bayes_classifier=0, alpha_value=1):

        self.classifier_name = "NaiveBayes"
        self.dataset = dataset

        # params
        if size < 0.0 or size > 1.0:
            self.size = 0.5
        else:
            self.size = size

        self.features = features
        self.naive_bayes_classifier = naive_bayes_classifier
        self.alpha_value = alpha_value

    def classification(self):
        # Naive Bayes classification process using the dataset requested
        data_label = data_label_extraction(self.dataset)

        if data_label == "Error":
            raise ValueError("Classification could not be done")

        data = data_label[0]
        labels = data_label[1]

        data = data_feature_selection(self.classifier_name, data, labels, self.dataset.value,
                                      self.features)

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.size, stratify=labels)

        if self.naive_bayes_classifier == 1:
            # Bernoulli classifier
            nb = BernoulliNB(alpha=self.alpha_value)

        else:
            # Gaussian classifier
            nb = GaussianNB()

        nb.fit(x_train, y_train)
        y_pred = nb.predict(x_test)

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
    def naive_bayes_classifier_getter(self):
        return self.naive_bayes_classifier

    @naive_bayes_classifier_getter.setter
    def naive_bayes_classifier_getter(self, new_naive_bayes_classifier):
        if new_naive_bayes_classifier not in (0, 1, "0", "1"):
            raise ValueError("Sorry, your value does not accomplish the criteria")
        self.naive_bayes_classifier = new_naive_bayes_classifier

    @property
    def alpha_value_getter(self):
        return self.alpha_value

    @alpha_value_getter.setter
    def alpha_value_getter(self, new_alpha_value):
        if type(new_alpha_value) == float and new_alpha_value > 0.0:
            self.alpha_value = new_alpha_value
        if type(new_alpha_value) == str:
            try:
                float(new_alpha_value)
                self.alpha_value = new_alpha_value
            except ValueError:
                raise ValueError("Sorry, your value does not accomplish the criteria")

        raise ValueError("Sorry, your value does not accomplish the criteria")
