from keystroke_program.classifiers.common_methods import data_label_extraction, data_feature_selection, performance_rates

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, dataset, size=0.2, features="no_feature_selection", activation=0,
                 hidden_layers_neurons=0, alpha=0.1):

        self.classifier_name = "Neural Network"
        self.dataset = dataset

        # params
        if size < 0.0 or size > 1.0:
            self.size = 0.5
        else:
            self.size = size

        self.features = features
        self.activation = activation
        self.hidden_layers_neurons = hidden_layers_neurons
        self.alpha = alpha

        self.activations = ["identity", "logistic", "tanh", "relu"]

    def classification(self):
        # Neural Network classification process using the dataset requested
        data_label = data_label_extraction(self.dataset)

        if data_label == "Error":
            raise ValueError("Classification could not be done")

        data = data_label[0]
        labels = data_label[1]

        data = data_feature_selection(self.classifier_name, data, labels, self.dataset.value,
                                      self.features)

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.size, stratify=labels)

        neural_network = MLPClassifier(activation=self.activation, solver="adam", alpha=self.alpha,
                                       hidden_layer_sizes=self.hidden_layers_neurons, max_iter=5000)

        neural_network.fit(x_train, y_train)
        y_pred = neural_network.predict(x_test)

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
    def activation_getter(self):
        return self.activation

    @activation_getter.setter
    def activation_getter(self, new_activation):
        if type(new_activation) == str:
            for i in range(len(self.activations)):
                if self.activations[i] == new_activation:
                    self.activation = i
        elif type(new_activation) == int and 0 <= new_activation <= 3:
            self.activation = new_activation

        raise ValueError("Sorry, your value does not accomplish the criteria")

    @property
    def hidden_layers_neurons_getter(self):
        return self.hidden_layers_neurons

    @hidden_layers_neurons_getter.setter
    def hidden_layers_neurons_getter(self, new_hidden_layers_neurons):
        if type(new_hidden_layers_neurons) == tuple or type(new_hidden_layers_neurons) == list:
            for j in new_hidden_layers_neurons:
                if type(new_hidden_layers_neurons[j]) == int:
                    continue

                elif type(new_hidden_layers_neurons[j]) == str and new_hidden_layers_neurons.isdigit():
                    new_hidden_layers_neurons[j] = int(new_hidden_layers_neurons)

                else:
                    raise ValueError("Sorry, your value does not accomplish the criteria")

        elif (type(new_hidden_layers_neurons) == str and new_hidden_layers_neurons.isdigit()) \
                or type(new_hidden_layers_neurons) == int:
            self.hidden_layers_neurons = int(new_hidden_layers_neurons)

        else:
            raise ValueError("Sorry, your value does not accomplish the criteria")

    @property
    def alpha_getter(self):
        return self.alpha

    @alpha_getter.setter
    def alpha_getter(self, new_alpha):
        try:
            self.alpha = float(new_alpha)
        except ValueError:
            raise ValueError("Sorry, your value does not accomplish the criteria")
