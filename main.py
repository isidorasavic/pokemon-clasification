import sys
import pandas as pd


def remove_suffixes(data):
    if pd.isna(data):
        return data
    if " kg" in data:
        return data.replace(" kg", "")
    if " m" in data:
        return data.replace(" m", "")
    if "%" in data:
        return data.replace("%", "")


def transform_data(dataset):
    dataset['secondary_type'] = dataset['secondary_type'].astype('category').cat.codes
    dataset['region'] = dataset['region'].astype('category').cat.codes
    dataset['category'] = dataset['category'].astype('category').cat.codes
    dataset['height'] = dataset['height'].apply(remove_suffixes)
    dataset['weight'] = dataset['weight'].apply(remove_suffixes)

    dataset['pokemon_family'] = dataset['pokemon_family'].astype('category').cat.codes

    # TODO: attack, defense i stamina su brojevi pa ih ne moramo menjati

    # TODO: cp-range i hp-range isprobati da li da se cuva pocetak i kraj intrvala u dve kolone,
    # ili samo sirina intervala u jednoj

    dataset['capture_rate'] = dataset['capture_rate'].apply(remove_suffixes)
    dataset['flee_rate'] = dataset['flee_rate'].apply(remove_suffixes)
    dataset['male_perc'] = dataset['male_perc'].apply(remove_suffixes)
    dataset['female_perc'] = dataset['female_perc'].apply(remove_suffixes)

    # todo: videti sta sa resistance, weakness

    dataset['wild_avail'] = dataset['wild_avail'].astype('category').cat.codes
    dataset['egg_avail'] = dataset['egg_avail'].astype('category').cat.codes
    dataset['raid_avail'] = dataset['raid_avail'].astype('category').cat.codes
    dataset['research_avail'] = dataset['research_avail'].astype('category').cat.codes
    dataset['shiny'] = dataset['shiny'].astype('category').cat.codes
    dataset['shadow'] = dataset['shadow'].astype('category').cat.codes

    # TODO: videti za pkedex_desc i poss_attacks

    X = dataset.drop("main_type", axis="columns")
    X = X.drop("number", axis="columns")
    X = X.drop("pic_url", axis="columns")

    dataset['main_type'] = dataset['main_type'].astype('category').cat.codes
    y = dataset["main_type"]
    return X, y


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    X, y = transform_data(df)

    print(X['poss_attacks'])

