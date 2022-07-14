import sys
import pandas as pd
from sklearn import metrics, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from colorama import init, Fore, Back, Style
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import KernelPCA
from sklearn.metrics import classification_report

init(autoreset=True)


def remove_suffixes(data):
    if pd.isna(data):
        return data
    if " kg" in data:
        return data.replace(" kg", "")
    if " m" in data:
        return data.replace(" m", "")
    if "%" in data:
        return data.replace("%", "")


def get_interval_beginning(data):
    if pd.isna(data):
        return data
    else:
        return data.split('-')[0].replace(",", ".")


def get_interval_ending(data):
    if pd.isna(data):
        return data
    else:
        return data.split('-')[1].replace(",", ".")


def transform_data(dataset):
    dataset['secondary_type'] = dataset['secondary_type'].astype('category').cat.codes
    dataset['region'] = dataset['region'].astype('category').cat.codes
    dataset['category'] = dataset['category'].astype('category').cat.codes
    dataset['height'] = dataset['height'].apply(remove_suffixes)
    dataset['weight'] = dataset['weight'].apply(remove_suffixes)

    dataset['pokemon_family'] = dataset['pokemon_family'].astype('category').cat.codes

    # attack, defense i stamina su brojevi pa ih ne moramo menjati

    # TODO: cp-range i hp-range isprobati da li da se cuva ovako ili sirina intervala
    dataset['cp_range_beginning'] = dataset['cp_range'].apply(get_interval_beginning)
    dataset['cp_range_ending'] = dataset['cp_range'].apply(get_interval_ending)

    dataset['hp_range_beginning'] = dataset['hp_range'].apply(get_interval_beginning)
    dataset['hp_range_ending'] = dataset['hp_range'].apply(get_interval_ending)

    dataset = dataset.drop("cp_range", axis="columns")
    dataset = dataset.drop("hp_range", axis="columns")

    dataset['capture_rate'] = dataset['capture_rate'].apply(remove_suffixes)
    median = dataset['capture_rate'].median()
    dataset['capture_rate'].fillna(median, inplace=True)

    dataset['flee_rate'] = dataset['flee_rate'].apply(remove_suffixes)
    median = dataset['flee_rate'].median()
    dataset['flee_rate'].fillna(median, inplace=True)

    dataset['male_perc'] = dataset['male_perc'].apply(remove_suffixes)
    median = dataset['male_perc'].median()
    dataset['male_perc'].fillna(median, inplace=True)

    dataset['female_perc'] = dataset['female_perc'].apply(remove_suffixes)
    median = dataset['female_perc'].median()
    dataset['female_perc'].fillna(median, inplace=True)

    # todo: videti sta sa resistance, weakness

    dataset['wild_avail'] = dataset['wild_avail'].astype('category').cat.codes
    dataset['egg_avail'] = dataset['egg_avail'].astype('category').cat.codes
    dataset['raid_avail'] = dataset['raid_avail'].astype('category').cat.codes
    dataset['research_avail'] = dataset['research_avail'].astype('category').cat.codes
    dataset['shiny'] = dataset['shiny'].astype('category').cat.codes
    dataset['shadow'] = dataset['shadow'].astype('category').cat.codes

    # TODO: videti za pkedex_desc

    X = dataset.drop("main_type", axis="columns")
    X = X.drop("number", axis="columns")
    X = X.drop("pic_url", axis="columns")

    # OBRISATI
    X = X.drop("pkedex_desc", axis="columns")
    X = X.drop("poss_attacks", axis="columns")
    X = X.drop("resistance", axis="columns")
    X = X.drop("weakness", axis="columns")
    X = X.drop("pokemon_name", axis="columns")

    dataset['main_type'] = dataset['main_type'].astype('category').cat.codes
    y = dataset["main_type"]
    return X, y


def kNN():
    neigh = KNeighborsClassifier(n_neighbors=29)
    neigh.fit(X_train, y_train)
    predicted_y = neigh.predict(X_test)
    print("K nearest neighbours: ", metrics.accuracy_score(y_test, predicted_y))


# najbolje radi ako se ne popunjavaju NaN polja (za sad bez lista obelezja)
def hist_boosting():
    model = HistGradientBoostingClassifier(random_state=8, learning_rate=0.09)
    model.fit(X_train, y_train)
    predicted_y = model.predict(X_test)

    print("Hist Gradient Boosting Classifier: ", metrics.accuracy_score(y_test, predicted_y))
    return predicted_y


def bagging():
    bag = BaggingClassifier(n_estimators=500, random_state=15)
    bag.fit(X_train, y_train)
    predicted_y = bag.predict(X_test)
    print("Bagging: ", metrics.accuracy_score(y_test, predicted_y))


# bas lose
def nb():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    predicted_y = gnb.predict(X_test)
    print("Naive bayes: ", metrics.accuracy_score(y_test, predicted_y))



if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    X, y = transform_data(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=8)



    # Kernel PCA
    k_pca = KernelPCA(n_components=19, kernel='rbf')
    # X_train = k_pca.fit_transform(X_train)
    # X_test = k_pca.fit_transform(X_test)

    # classifier = LogisticRegression(random_state=8)
    # classifier.fit(X_train, y_train)
    #
    # hist_boosting()
    #
    y_pred = hist_boosting()

    print(metrics.accuracy_score(y_test, y_pred))


    # for i in range(len(expected_y.values)):
    #     if expected_y.values[i] == predicted_y[i]:
    #         text = Fore.GREEN + "Expected: " + str(expected_y.values[i]) + "  Predicted: "+str(predicted_y[i]) + Fore.RESET
    #     else:
    #         text = Fore.RED + "Expected: " + str(expected_y.values[i]) + "  Predicted: "+str(predicted_y[i]) + Fore.RESET
    #     print(text)
