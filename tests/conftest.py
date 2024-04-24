import pytest
import pandas as pd


@pytest.fixture
def df_taxa_table():
    return pd.read_csv("tests/data/taxa_table.csv")


@pytest.fixture
def df_taxa_table_no_labels():
    return pd.read_csv("tests/data/taxa_table_no_labels.csv")


@pytest.fixture
def df_leaf_in_cols():
    return pd.read_csv("tests/data/leaf_in_cols.csv")


@pytest.fixture
def df_simple_in_cols():
    return pd.read_csv("tests/data/simple_in_cols.csv")


@pytest.fixture
def df_ancestor_as_label():
    return pd.read_csv("tests/data/ancestor_as_label.csv")


@pytest.fixture
def df_reoccuring_value():
    return pd.read_csv("tests/data/reoccuring_value.csv")


@pytest.fixture
def df_large_taxa_table():
    return pd.read_csv("tests/data/large_taxa_table.csv")


@pytest.fixture
def df_all_taxa_table():
    return pd.read_csv("tests/data/all_taxa_table.csv")


@pytest.fixture
def df_full_taxa_table():
    return pd.read_csv("tests/data/full_taxa_table.csv")


@pytest.fixture
def df_simple_table():
    return pd.read_csv("tests/data/simple_table.csv")


@pytest.fixture
def df_very_simple():
    return pd.read_csv("tests/data/very_simple.csv")


@pytest.fixture
def df_all_gaps():
    return pd.read_csv("tests/data/all_gaps.csv")


@pytest.fixture
def df_fillable_gaps():
    return pd.read_csv("tests/data/fillable_gaps.csv")


@pytest.fixture
def df_fillable_gaps_not():
    return pd.read_csv("tests/data/fillable_gaps_not.csv")


@pytest.fixture
def df_presence():
    return pd.read_csv("tests/data/presence.csv")


@pytest.fixture
def dict_abb_map():
    with open("tests/data/abb_map.txt", "r") as f:
        content = f.readlines()
    c = [x.strip().split(",") for x in content]
    return {x[0]: x[1] for x in c}
