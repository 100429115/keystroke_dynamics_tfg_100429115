# TRABAJO FIN DE GRADO: AUTENTICACIÓN BIOMÉTRICA BASADA EN PULSACIONES DE TECLAS
# FINAL DEGREE DISSERTATION: BIOMETRIC AUTHENTICATION BASED ON KEYSTROKES

# NAME: ADRIÁN SERRANO NAVARRO
# GRADO EN INGENIERÍA INFORMÁTICA, UNIVERSIDAD CARLOS III DE MADRID
# COMPUTER ENGINEERING DEGREE, CARLOS III OF MADRID UNIVERSITY

from dataset.dataset import Dataset
from classifiers.knn import KNN
from classifiers.naive_bayes import NaiveBayes
from classifiers.svm import SVM
from classifiers.random_forest import RandomForest
from classifiers.neural_network import NeuralNetwork


if __name__ == '__main__':

    # Request of the problem parameters, dataset and classifier
    # Each print contains, the values already proved.

    print("IN ORDER TO START WITH THE TEST")

    print("Select the dataset by their given numbers: ")
    dataset_value = int(input("dns_2009=1, greyc_nyslab=2, mobikey=3, mobikey temporal=4:"))

    # Creation and preparation of the requested dataset
    dataset = Dataset(dataset_value)
    dataset.dataset_modeling()

    print("Select the classifiers by their given numbers:")
    classifier_name = int(input("KNN=1, Naive Bayes=2, Random Forest=3, SVM=4, Neural Network=5: "))

    print("\nSelect the parameters of the classifiers: ", classifier_name)

    print("proven sizes = [0.2, 0.4]")
    size = float(input("Select test size: "))

    print("proven feature selection techniques:\n"
          "# feature selection dns 2009 = [no_feature_selection, PCA4, PCA15, PCA20, "
          "SFS, escalado_no_feature_selection, escalado_SFS]\n"

          "# feature selection greyc-nislab = [no_feature_selection, PCA13, PCA15, PCA17, "
          "SFS, escalado_no_feature_selection, escalado_SFS]\n"

          "feature selection mobikey = [no_feature_selection, PCA90, PCA95, PCA103, "
          "SFS, escalado_no_feature_selection, escalado_SFS]\n"
        
          "feature selection mobikey temporal = [no_feature_selection, escalado_no_feature_selection, SFS,"
          "escalado_SFS]\n")

    features = input("Select data features selection technique, in the format shown: ")

    # Selection of parameters depending on the requested algorithm

    if classifier_name == 1:
        # KNN
        print("proven neighbors = [1, 3, 5, 7, 10]")
        print("proven distances = [minkowski, manhattan, canberra, l1, cosine]")
        print("proven weights = [uniform, distance]")

        neighbors = int(input("Introduce neighbors: "))
        metric = input("Introduce metric: ")
        weight = input("Introduce weight: ")

        # calling KNN classifier
        classifier = KNN(dataset, size, features, neighbors, metric, weight)

    elif classifier_name == 2:
        # Naive Bayes
        print("Proven Naive Bayes classifier, naive_bayes_classifier + alpha, "
              "gaussian, BernoulliNB_1, BernoulliNB_0,BernoulliNB_5, BernoulliNB_20")

        naive_bayes_classifier = int(input("Introduce classifier, Gaussian(0), Bernoulli(1): "))
        alpha_value = int(input("introduce alpha value, 0 if gaussian, any int for Bernoulli: "))

        # calling Naive-Bayes classifier
        classifier = NaiveBayes(dataset, size, features, naive_bayes_classifier, alpha_value)

    elif classifier_name == 3:
        # Random Forest
        print("Proven Random Forest criterion, entropy, gini")
        criterion = int(input("Introduce criterion, entropy(0), gini(1): "))

        print("Proven DNS 2009 Random Forest depth [13, 15, 17]")
        print("Proven Greyc-nislab Random Forest depth [13, 15, 17]")
        print("Proven Mobikey Random Forest depth [17, 20, 23]")
        print("Proven Mobikey temporal Random Forest depth [7, 10, 13]")
        depth = int(input("Introduce depth: "))

        print("Proven DNS 2009 Random Forest max characteristics number (5, 7, sqrt)")
        print("Proven Greyc-nislab Random Forest max characteristics number (3, 5, sqrt)")
        print("Proven Mobikey Random Forest max characteristics number (3, 5, sqrt)")
        print("Proven Mobikey temporal Random Forest max characteristics number (9, 11, sqrt)")
        max_characteristics_number = input("Introduce number of characteristics: ")

        print("Proven Random Forest DNS 2009, Greyc-nislab, Mobikey estimators number [175, 200, 225]")
        print("Proven Mobikey temporal Random Forest estimators number [25, 50, 75]")
        estimators = int(input("Introduce estimators: "))

        # calling Random Forest classifier
        classifier = RandomForest(dataset, size, features, criterion, depth, max_characteristics_number, estimators)

    elif classifier_name == 4:
        # Support vector machine (SVM)
        print("Proven SVM Kernels linear(0), rbf(1), sigmoid(2), precomputed(3)")
        kernel = int(input("Introduce kernel: "))

        print("Proven Dns 2009 SVM C value [1.0, 88.58667904100814, 100.0]")
        print("Proven Greyc-nislab SVM C value [1.0, 46.41588833612782, 100.0]")
        print("Proven Mobikey SVM C value [1.0, 10.0, 46.41588833612782, 100.0]")
        print("Proven Mobikey temporal SVM C value [10000.0, 100000.0, 500000.0]")
        c = float(input("Introduce C value: "))

        print("Proven SVM DNS 2009, Greyc-nislab, Mobikey gamma value (1.0, 100.0, scale, auto)")
        print("Proven SVM Mobikey temporal gamma value (5000.0, scale, auto)")
        gamma = input("Introduce gamma value: ")

        # calling SVM classifier
        classifier = SVM(dataset, size, features, kernel, c, gamma)

    else:
        # Neural Network
        # solver = adam, iterations=5000, learning rate init=0.001

        print("Proven Neural Network activation technique identity(0), logistic(1), tanh(2), relu(3)")
        activation = int(input("Introduce activation: "))

        print("Proven Dns 2009 hidden layers [100, 250]")
        print("Proven Greyc-nislab hidden layers [500, (500, 500)]")
        print("Proven Mobikey hidden layers [(1000, 1000), (700, 700)]")
        print("Proven Mobikey temporal hidden layers [50, 100, 300]")
        hidden_layers = int(input("Introduce hidden layers number: "))
        if hidden_layers > 0:
            hidden_layers_neurons = []
            for i in range(hidden_layers):
                print("Introduce neurons of layer ", i+1)
                neurons = int(input("number of neurons: "))
                hidden_layers_neurons.append(neurons)
        else:
            hidden_layers_neurons = 0

        print("Proven Dns 2009 alpha value [0.004641588833612777, 2.15]")
        print("Proven Greyc-nislab alpha value [0.001, 2.15]")
        print("Proven Mobikey alpha value [0.1, 2.15]")
        print("Proven Mobikey Temporal alpha value [0.1, 10.0, 20.0]")
        alpha = float(input("Introduce alpha value: "))

        # calling neural network classifier
        classifier = NeuralNetwork(dataset, size, features, activation, tuple(hidden_layers_neurons), alpha)

    # Presentation of the results
    print("\nResult:", classifier.classification())
