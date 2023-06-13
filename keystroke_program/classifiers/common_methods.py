from sklearn.preprocessing import StandardScaler
import statistics as stats

from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


def data_label_extraction(dataset):
    # Splitting of labels and typing data depending on the dataset
    if dataset.value == 1:
        # DNS2009
        x = dataset.dataframe.drop(["subject", "sessionIndex", "rep"], axis="columns")
        y = dataset.dataframe.subject

    elif dataset.value == 2:
        # Greyc-nislab
        x = dataset.dataframe.drop(["User_ID", "Gender", "Age", "Handedness", "Password", "Class",
                                    "Keystroke Template Vector"], axis="columns")
        y = dataset.dataframe.User_ID

    elif dataset.value == 3:
        # Mobikey
        x = dataset.dataframe.drop(["UserId"], axis="columns")
        y = dataset.dataframe.UserId

    elif dataset.value == 4:
        # Mobikey temporal
        x = dataset.dataframe.drop(["UserId"], axis="columns")
        y = dataset.dataframe.UserId

    else:
        # Other dataset
        if len(dataset.dataframe.columns) >= 2:
            x = dataset.dataframe.drop(dataset.dataframe.columns[0], axis="columns")
            y = dataset.dataframe.iloc[:, 0]

        else:
            return "Error"

    return x, y


def data_feature_selection(classifier, data, labels, data_value, feature_selection,
                           n_neighbors=1, metric="manhattan", weight="uniform"):

    # preprocessing of the dataset, standardisation and dimensionality reduction, PCA and SFS
    ss = StandardScaler()

    if "escalado" in feature_selection:
        # scaled values
        data = ss.fit_transform(data)

    if "PCA" in feature_selection:
        components = feature_selection[3:]
        pca2 = PCA(n_components=int(components))
        data = pca2.fit_transform(ss.fit_transform(data))

    elif "SFS" in feature_selection:
        # the predictor to use depends on the dataset
        if classifier == "KNN":
            sfs = SequentialFeatureSelector(KNeighborsClassifier(
                                                        n_neighbors=n_neighbors, metric=metric, weights=weight),
                                            n_features_to_select=len(data.columns) // 2)
        else:
            # best KNN neighbors value is used
            if data_value == 1:
                # DNS2009
                sfs = SequentialFeatureSelector(KNeighborsClassifier(
                                                        n_neighbors=7, metric="manhattan", weights="distance"),
                                                n_features_to_select=len(data.columns) // 2)
            elif data_value == 2:
                # Greyc-nislab
                sfs = SequentialFeatureSelector(KNeighborsClassifier(
                                                        n_neighbors=3, metric="canberra", weights="distance"),
                                                n_features_to_select=len(data.columns) // 2)
            elif data_value == 3:
                # Mobikey
                sfs = SequentialFeatureSelector(KNeighborsClassifier(
                                                        n_neighbors=5, metric="canberra", weights="distance"),
                                                n_features_to_select=len(data.columns) // 2)

            elif data_value == 4:
                # Mobikey temporal
                sfs = SequentialFeatureSelector(KNeighborsClassifier(
                                                        n_neighbors=1, metric="cosine", weights="uniform"),
                                                n_features_to_select=len(data.columns) // 2)
            else:
                sfs = SequentialFeatureSelector(KNeighborsClassifier(),
                                                n_features_to_select=len(data.columns) // 2)
        sfs.fit(data, labels)
        data = sfs.transform(data)

    return data


def performance_rates(y_test, y_pred):

    # it return the performance rates, f1, accuracy, TFP and TFN
    f1 = stats.mean(f1_score(y_test, y_pred, average=None))
    accuracy = accuracy_score(y_test, y_pred)

    matrix = confusion_matrix(y_test, y_pred)
    n_users = n_columns = matrix.shape[1]

    fp_total = 0
    fn_total = 0

    # u represents each of the users
    for u in range(n_users):

        # false positives
        fp = 0

        # true negatives
        tn = 0

        # false negatives
        fn = 0

        # true positives
        tp = 0

        for r in range(matrix.shape[0]):
            for c in range(n_columns):
                # For clarity, each metric is calculated separately
                if r == u and c == u:
                    tp = matrix[r][c]
                if r != u and c != u:
                    tn = tn + matrix[r][c]
                if r == u and c != u:
                    fp = fp + matrix[r][c]
                if r != u and c == u:
                    fn = fn + matrix[r][c]

        if fp == 0 and tn == 0:
            user_fp_rate = 0

        else:
            user_fp_rate = fp / (fp + tn)

        fp_total += user_fp_rate

        if fn == 0 and tp == 0:
            user_fn_rate = 0

        else:
            user_fn_rate = fn / (fn + tp)

        fn_total += user_fn_rate

    mean_fp = fp_total / n_users
    mean_fn = fn_total / n_users

    return f1, accuracy, mean_fp, mean_fn
