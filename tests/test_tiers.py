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
    assert (
        "Trying to get an nonexistent node aldsk. Perhaps you are trying to find a label?"
        in str(excinfo.value)
    )
    tree.show()
    tree.show_simple()
    tree.show(labels=True)
    with pytest.raises(ValueError) as excinfo:
        tree.show_simple(labels=True)
    assert "Labels are not supported in simple tree" in str(excinfo.value)

    tree.set_level("genus")
    assert tree.level == "05_genus"
    assert tree.map("RaBa") == "Radix"
    assert tree.map(["Asellus_aquaticus", "Caenis_horaria"]) == ["Asellus", "Caenis"]
    assert tree.map(
        ["As", "Asellus_aquaticus", "Oulimnius_tuberculatus"], level="type"
    ) == ["Asellus", "Asellus aquaticus", "Oulimnius tuberculatus larva"]
    assert tree.map("Oulimnius_tuberculatus") == "Oulimnius"

    assert tree.map(
        ["As", "Asellus_aquaticus", "Oulimnius_tuberculatus"], level="type", strict=True
    ) == [None, None, "Oulimnius tuberculatus larva"]

    assert tree.map(["Asellus", "Caenis"], level="phylum", nodes=True) == [
        "Arthropoda",
        "Arthropoda",
    ]
    assert tree.map(["Caenis", "As"], level="phylum") == ["Arthropoda", "Arthropoda"]
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
    assert "label 'Asellidae' is in the hierarchy as a higher-level node." in str(
        record[0].message
    )

    tree = tiers.Tree.from_dataframe(df_ancestor_as_label, node_remapping=True)
    tree.show_simple()
    assert tree.map("Asellus") == "Asellus"


@pytest.mark.usefixtures("df_reoccuring_value")
def test_reoccuring_value(df_reoccuring_value):
    """Test the creation of a tree from a dataframe with a reoccuring value in the columns."""
    with pytest.raises(ValueError) as excinfo:
        tree = tiers.Tree.from_dataframe(df_reoccuring_value)
    assert "Columns are not disjoint. {'Isopoda'} exist multiple times" in str(
        excinfo.value
    )


@pytest.mark.usefixtures("df_all_taxa_table")
def test_gap_filling(df_all_taxa_table):
    with pytest.warns(UserWarning) as record:
        tree = tiers.Tree.from_dataframe(df_all_taxa_table)
    assert len(record) == 1
    assert "Dataframe had gaps in rows" in str(record[0].message)
    tree = tiers.Tree.from_dataframe(df_all_taxa_table, fill_gaps=True)


@pytest.mark.usefixtures("df_fillable_gaps_not")
def test_fillable_gaps_not(df_fillable_gaps_not):
    with pytest.raises(ValueError) as excinfo:
        tree = tiers.Tree.from_dataframe(df_fillable_gaps_not, fill_gaps=True)
    assert "The dataframe has duplicate name-parent pairs." in str(excinfo.value)


@pytest.mark.usefixtures("df_simple_table")
def test_update_labels(df_simple_table):
    """Tests the tree.labels() method."""
    tree = tiers.Tree.from_dataframe(df_simple_table)
    old_map = tree.label_map
    tree.update_label_map({"As": "Asellus", "Ca": "Caenis"})
    # This should not change the tree, only return a new one
    assert old_map == tree.label_map

    # This returns a new tree with a changed label map
    tree = tree.update_label_map({"As": "Asellus", "Ca": "Caenis"})

    # This should raise an error
    with pytest.raises(ValueError) as excinfo:
        tree = tree.update_label_map({"test_label": "test_node"})
    assert "Node 'test_node' not in nodes" in str(excinfo.value)

    # Trying to update a label to a higher-level node should raise a warning
    with pytest.warns(UserWarning) as record:
        tree = tree.update_label_map({"Asellus": "Asellus aquaticus"})
    assert "label 'Asellus' is in the hierarchy as a higher-level node." in str(
        record[0].message
    )
    tree = tree.update_label_map({"Asellus": "Asellus aquaticus"}, node_remapping=True)


@pytest.mark.usefixtures("df_simple_table")
def test_labels(df_simple_table):
    tree = tiers.Tree.from_dataframe(df_simple_table)
    tree = tree.update_label_map({"AsAq": "Asellus aquaticus", "As": "Asellus"})
    labels = tree.labels(["Asellus aquaticus", "Insecta", "Caenis horaria"])
    assert labels == [["Asellus aquaticus", "AsAq"], [], ["Caenis_horaria"]]

    assert tree.labels("Asellus aquaticus") == ["Asellus aquaticus", "AsAq"]
    assert tree.labels("asd") == []  # non-existing nodes should return an empty list


@pytest.mark.usefixtures("df_very_simple")
def test_rels(df_very_simple):
    rel_ref = pd.DataFrame(
        {
            "names": ["g1", "A", "g2", "B", "g3"],
            "parents": ["A", "root", "B", "root", "B"],
        }
    )
    tree = tiers.Tree.from_dataframe(df_very_simple)
    pd.testing.assert_frame_equal(tree.rel, rel_ref)
    pd.testing.assert_frame_equal(tree.relation_table, rel_ref)

    assert tree.rel_dict == {"g1": "A", "A": "root", "g2": "B", "B": "root", "g3": "B"}
    assert tree.parent_map == {
        "g1": "A",
        "A": "root",
        "g2": "B",
        "B": "root",
        "g3": "B",
    }


@pytest.mark.usefixtures("df_full_taxa_table", "dict_abb_map")
def test_extend_labels(df_full_taxa_table, dict_abb_map):
    tree = tiers.Tree.from_dataframe(df_full_taxa_table)
    abb_map = {k: v for k, v in dict_abb_map.items() if v in tree.nodes}
    tree = tree.update_label_map(abb_map)

    assert tree.extend_label("Oligochaeta") == [
        "Animalia",
        "Angiospermae",
        "Dicotyledoneae",
        "Campanulales",
        "Asteraceae",
        "Oligochaeta",
        "",
        "",
        "Oligochaeta",
    ]
    assert tree.extend_labels(["Oligochaeta"], return_string=True, pad=True) == [
        "Animalia - Angiospermae - Dicotyledoneae - Campanulales - Asteraceae - Oligochaeta -  -  - Oligochaeta"
    ]

    assert tree.extend_labels(
        ["Ol", "OuTu"], levels=["order", "genus"], return_string=True, pad=True
    ) == ["Campanulales - Oligochaeta -   Ol", "  Coleoptera -   Oulimnius - OuTu"]
