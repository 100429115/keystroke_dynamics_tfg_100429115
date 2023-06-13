from keystroke_program.classifiers.common_methods import data_label_extraction, data_feature_selection, performance_rates
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

kernels = ["linear", "rbf", "sigmoid", "precomputed"]

class SVM:

    def __init__(self, dataset, size=0.2, features="no_feature_selection", kernel=1, c=1.0, gamma="1.0"):

        self.classifier_name = "SVM"
        self.dataset = dataset

        # params
        if size < 0.0 or size > 1.0:
            self.size = 0.5
        else:
            self.size = size

        self.features = features
        self.kernel = kernel
        self.c = c
        if gamma != "scale" and gamma != "auto":
            self.gamma = float(gamma)
        else:
            self.gamma = gamma

    def classification(self):
        # SVM classification process using the dataset selected
        data_label = data_label_extraction(self.dataset)

        if data_label == "Error":
            raise ValueError("Classification could not be done")

        data = data_label[0]
        labels = data_label[1]

        data = data_feature_selection(self.classifier_name, data, labels, self.dataset.value,
                                      self.features)

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.size, stratify=labels)

        svc = SVC(C=self.c, gamma=self.gamma, kernel=kernels[self.kernel])

        if self.kernel == 3:
            gram_train = np.matmul(np.array(x_train), np.array(x_train).T)
            gram_test = np.matmul(np.array(x_test), np.array(x_train).T)
            svc.fit(gram_train, y_train)
            y_pred = svc.predict(gram_test)

        else:
            svc.fit(x_train, y_train)
            y_pred = svc.predict(x_test)

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
    def kernel_getter(self):
        return self.kernel

    @kernel_getter.setter
    def kernel_getter(self, new_kernel):
        kernels = ("linear", "poly", "rbf", "sigmoid", "precomputed")
        if type(new_kernel) == str:
            if new_kernel.isdigit() and 0 <= int(new_kernel) <= 3:
                self.kernel = int(new_kernel)

            else:
                for i in range(len(kernels)):
                    if kernels[i] == new_kernel:
                        self.kernel = i

        elif type(new_kernel) == int and 0 <= new_kernel <= 3:
            self.kernel = new_kernel
        raise ValueError("Sorry, your value does not accomplish the criteria")

    @property
    def c_getter(self):
        return self.c

    @c_getter.setter
    def c_getter(self, new_c):
        if (type(new_c) == float or type(new_c) == int) and new_c > 0.0:
            self.c = float(new_c)

        elif type(new_c) == str:
            try:
                self.c = float(new_c)

            except ValueError:
                raise ValueError("Sorry, your value does not accomplish the criteria")

        raise ValueError("Sorry, your value does not accomplish the criteria")

    @property
    def gamma_getter(self):
        return self.gamma

    @gamma_getter.setter
    def gamma_getter(self, new_gamma):
        if type(new_gamma) == str and new_gamma in ["scale", "auto"]:
            self.gamma = new_gamma

        elif (type(new_gamma) == float or type(new_gamma) == int) and new_gamma > 0.0:
            self.gamma = float(new_gamma)

        raise ValueError("Sorry, your value does not accomplish the criteria")
