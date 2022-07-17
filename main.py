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
import json

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


def get_interval_size(data):
    if pd.isna(data):
        return data
    else:
        return float(data.split('-')[0].replace(",", ".")) - float(data.split('-')[1].replace(",", "."))


def add_values_in_dict(sample_dict, key, list_of_values):
    ''' Append multiple values to a key in
        the given dictionary '''
    if key not in sample_dict:
        sample_dict[key] = list()
    sample_dict[key].extend(list_of_values)
    return sample_dict


def attacks_categoristaion(dataset):
    ''' Function for poss_attacks categorisation
        Input: dataset
        Output dataset with cahnged values to categorised'''

    # Creating sorted list of unique attack words
    all_attacks = dataset["poss_attacks"]
    all = ''
    for attack in all_attacks:
        all = all + attack

    all = all.replace('][', ' ')
    complete = ''.join(ch for ch in all if ch.isalnum() or ch == ' ')
    final_list = complete.lstrip().rstrip().strip().split(" ")
    dict_of_counts = {item: final_list.count(item) for item in final_list}
    sort = {k: v for k, v in sorted(dict_of_counts.items(), key=lambda item: item[1])}

    # Manually removed words that are misleading
    keys_to_delete = ['Beam', 'Sludge', 'Ball', 'Attack', 'Tackle', 'Bite', 'Pulse', 'Punch', 'Bomb', 'Power']
    for k in keys_to_delete:
        if k in sort.keys():
            del sort[k]

    # Find values in data set and replace "list" strings whit one unique string
    for ind in dataset.index:
        value = dataset['poss_attacks'][ind]
        value_fix = ''.join(ch for ch in value if ch.isalnum() or ch == ' ')
        next = ''

        # List is sorted, so we always get the word with the highest value at the end
        for k in sort.keys():
            if k in value_fix:
                next = k
        next = next.strip()
        next = next.rstrip()
        next = next.lstrip()

        # If empty string, handle
        if next == '':
            next = 'Def'

        dataset.at[ind, 'poss_attacks'] = next

    # Strings to number
    dataset['poss_attacks'] = dataset['poss_attacks'].astype('category').cat.codes

    return dataset

def transform_data(dataset):
    dataset['secondary_type'] = dataset['secondary_type'].astype('category').cat.codes
    dataset['region'] = dataset['region'].astype('category').cat.codes
    dataset['category'] = dataset['category'].astype('category').cat.codes
    dataset['height'] = dataset['height'].apply(remove_suffixes)
    dataset['weight'] = dataset['weight'].apply(remove_suffixes)

    dataset['pokemon_family'] = dataset['pokemon_family'].astype('category').cat.codes

    # ako zelimo sve podatke o intervalima
    dataset['cp_range_beginning'] = dataset['cp_range'].apply(get_interval_beginning)
    dataset['cp_range_ending'] = dataset['cp_range'].apply(get_interval_ending)

    dataset['hp_range_beginning'] = dataset['hp_range'].apply(get_interval_beginning)
    dataset['hp_range_ending'] = dataset['hp_range'].apply(get_interval_ending)

    # ako zelimo samo sirinu intervala
    # dataset['cp'] = dataset['cp_range'].apply(get_interval_size)
    # dataset['hp'] = dataset['hp_range'].apply(get_interval_size)

    # uvek brisanje starih kolona
    dataset = dataset.drop("cp_range", axis="columns")
    dataset = dataset.drop("hp_range", axis="columns")

    # dole zakomentarisano je popunjavanje praznih polja, sto ako se radi, dobijaju se za nijansu losiji rezultati
    dataset['capture_rate'] = dataset['capture_rate'].apply(remove_suffixes)
    # median = dataset['capture_rate'].median()
    # dataset['capture_rate'].fillna(median, inplace=True)

    dataset['flee_rate'] = dataset['flee_rate'].apply(remove_suffixes)
    # median = dataset['flee_rate'].median()
    # dataset['flee_rate'].fillna(median, inplace=True)

    dataset['male_perc'] = dataset['male_perc'].apply(remove_suffixes)
    # median = dataset['male_perc'].median()
    # dataset['male_perc'].fillna(median, inplace=True)

    dataset['female_perc'] = dataset['female_perc'].apply(remove_suffixes)
    # median = dataset['female_perc'].median()
    # dataset['female_perc'].fillna(median, inplace=True)

    dataset['wild_avail'] = dataset['wild_avail'].astype('category').cat.codes
    dataset['egg_avail'] = dataset['egg_avail'].astype('category').cat.codes
    dataset['raid_avail'] = dataset['raid_avail'].astype('category').cat.codes
    dataset['research_avail'] = dataset['research_avail'].astype('category').cat.codes
    dataset['shiny'] = dataset['shiny'].astype('category').cat.codes
    dataset['shadow'] = dataset['shadow'].astype('category').cat.codes
    dataset['weakness'] = dataset['weakness'].astype('category').cat.codes
    dataset['resistance'] = dataset['resistance'].astype('category').cat.codes

    # Custom categorisation
    dataset = attacks_categoristaion(dataset)

    X = dataset.drop("main_type", axis="columns")
    X = X.drop("pokemon_name", axis="columns")
    X = X.drop("number", axis="columns")
    X = X.drop("pic_url", axis="columns")
    # X = X.drop("poss_attacks", axis="columns")

    X = X.drop("pkedex_desc", axis="columns")

    dataset['main_type'] = dataset['main_type'].astype('category').cat.codes
    y = dataset["main_type"]
    return X, y


# osrednje
def kNN():
    neigh = KNeighborsClassifier(n_neighbors=29)
    neigh.fit(X_train, y_train)
    predicted_y = neigh.predict(X_test)
    print("K nearest neighbours: ", metrics.accuracy_score(y_test, predicted_y))


# najbolje radi
def hist_boosting():
    model = HistGradientBoostingClassifier(random_state=8, learning_rate=0.09)
    model.fit(X_train, y_train)
    predicted_y = model.predict(X_test)

    print("Hist Gradient Boosting Classifier: ", metrics.accuracy_score(y_test, predicted_y))
    return predicted_y


# osrednje radi
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


def print_results(predicted, expected):
    for i in range(len(expected.values)):
        if expected.values[i] == predicted[i]:
            text = Fore.GREEN + "Expected: " + str(expected.values[i]) + "  Predicted: " + str(
                predicted[i]) + Fore.RESET
        else:
            text = Fore.RED + "Expected: " + str(expected.values[i]) + "  Predicted: " + str(predicted[i]) + Fore.RESET
        print(text)


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    X, y = transform_data(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=8)

    predicted_y = hist_boosting()

    print_results(predicted_y, y_test)
