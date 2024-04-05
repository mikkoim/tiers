
import pytest
import tiers
import pandas as pd

@pytest.mark.usefixtures("df_taxa_table")
def test_tree_creation(df_taxa_table):
    """Test the creation of a tree from a dataframe."""
    tree = tiers.Tree.from_dataframe(df_taxa_table)
    assert tree is not None
    assert tree.map("As") == "Asellus"
    assert tree.map("Caenis") == "Caenis"
    with pytest.raises(KeyError) as excinfo:
        tree.map("aldsk")
    assert 'Trying to get an nonexistent node aldsk. Perhaps you are trying to find a label?' in str(excinfo.value)
    tree.show()
    tree.show_simple()
    
    tree.set_level("genus")
    assert tree.level == '05_genus'
    assert tree.map("RaBa") == 'Radix'
    assert tree.map(["Asellus_aquaticus", "Caenis_horaria"]) == ["Asellus", "Caenis"]
    assert tree.map(["As",
          "Asellus_aquaticus",
          "Oulimnius_tuberculatus"],
          level="type") == ['Asellus', 'Asellus aquaticus', 'Oulimnius tuberculatus larva']
    assert tree.map("Oulimnius_tuberculatus") == 'Oulimnius'

    assert tree.map(["As",
          "Asellus_aquaticus",
          "Oulimnius_tuberculatus"],
          level="type",
          strict=True) == [None, None, 'Oulimnius tuberculatus larva']

    assert tree.map(["Asellus", "Caenis"], level="phylum", nodes=True) == ['Arthropoda', 'Arthropoda']
    assert tree.map(["Caenis", "As"], level="phylum") == ['Arthropoda', 'Arthropoda']
    assert tree.get_level("As") == "05_genus"
    assert tree.get_level("As", prefix=False) == "genus"

@pytest.mark.usefixtures("df_taxa_table")
def test_matches_and_ancestors(df_taxa_table):
    tree = tiers.Tree.from_dataframe(df_taxa_table)
    assert tree.in_ancestors("Caenis horaria", "Caenis")
    assert not tree.in_ancestors("Caenis horaria", "Caenis horaria")
    assert tree.match("Caenis horaria", "Caenis")
    assert tree.match("Caenis", "Caenis horaria")
    assert not tree.match("Baetidae", "Caenis horaria")
    assert not tree.match("Caenis horaria", "Baetidae")
    assert tree.match("Caenis horaria", "Caenis horaria")
    assert tree.lca("Caenis horaria", "Baetidae") == "Ephemeroptera"
    assert tree.lca("Caenis_horaria", "As") == "Arthropoda"
    assert tree.match_level("Caenis horaria", "Baetidae") == "03_order"

@pytest.mark.usefixtures("df_leaf_in_cols")
def test_leaf_in_cols(df_leaf_in_cols):
    """Test the creation of a tree from a dataframe with a column named 'leaf',
    which is not allowed"""
    with pytest.raises(ValueError) as excinfo:
        tree = tiers.Tree.from_dataframe(df_leaf_in_cols)
    assert "A level column cannot be named 'leaf'" in str(excinfo.value)

@pytest.mark.usefixtures("df_simple_in_cols")
def test_simple_in_cols(df_simple_in_cols):
    """Test the creation of a tree from a dataframe with a column named 'simple',
      which is not allowed"""
    with pytest.raises(ValueError) as excinfo:
        tree = tiers.Tree.from_dataframe(df_simple_in_cols)
    assert "A level column cannot be named 'simple'" in str(excinfo.value)

@pytest.mark.usefixtures("df_ancestor_as_label")
def test_ancestor_as_label(df_ancestor_as_label):
    """Test the creation of a tree from a dataframe with an ancestor as a label.
      This is only allowed when node_remapping is True."""
    with pytest.warns(UserWarning) as record:
        tree = tiers.Tree.from_dataframe(df_ancestor_as_label, node_remapping=False)
    assert len(record) == 1
    assert "label 'Asellidae' is in the hierarchy as a higher-level node." in str(record[0].message)

    tree= tiers.Tree.from_dataframe(df_ancestor_as_label, node_remapping=True)
    tree.show_simple()
    assert tree.map("Asellus") == 'Asellus'

@pytest.mark.usefixtures("df_reoccuring_value")
def test_reoccuring_value(df_reoccuring_value):
    """Test the creation of a tree from a dataframe with a reoccuring value in the columns."""
    with pytest.raises(ValueError) as excinfo:
        tree = tiers.Tree.from_dataframe(df_reoccuring_value)
    assert "Columns are not disjoint. {'Isopoda'} exist multiple times" in str(excinfo.value)


@pytest.mark.usefixtures("df_large_taxa_table")
def test_large_table_with_gaps(df_large_taxa_table):
    """Test the creation of a tree from a large dataframe with gaps in the rows."""
    with pytest.warns(UserWarning) as record:
        tree = tiers.Tree.from_dataframe(df_large_taxa_table)
    assert len(record) == 1
    assert "Dataframe had gaps in rows" in str(record[0].message)

@pytest.mark.usefixtures("df_all_taxa_table")
def test_gap_filling(df_all_taxa_table):
    with pytest.warns(UserWarning) as record:
        tree = tiers.Tree.from_dataframe(df_all_taxa_table)
    assert len(record) == 1
    assert "Dataframe had gaps in rows" in str(record[0].message)
