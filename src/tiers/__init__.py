"""
Tiers - a hierarchical label handling library
"""

import bigtree
import pandas as pd
import warnings
import numpy as np
from typing import Union
from collections.abc import Iterable


def columns_disjoint(df):
    """Checks if dataframe columns are disjoint from each other"""
    s = set()
    for col in df.columns:
        s2 = set(df[col].dropna().unique().tolist())
        if s.isdisjoint(s2):
            s = s.union(s2)
        else:
            raise ValueError(
                f"Columns are not disjoint. {s.intersection(s2)} exist "
                "multiple times"
            )
    return True


def get_row_leaf(row):
    "Returns the leftmost not-NaN value of a pandas row"
    for i in range(len(row)):
        if pd.isna(row.iloc[i]):
            return row.iloc[i - 1]
    return row.iloc[i]


def get_leaves(df) -> pd.Series:
    """Find the leaves (leftmost not-NaN value of each row) for a dataframe."""
    # Check for gaps
    gaps = check_for_gaps(df)
    if gaps:
        raise ValueError(
            f"Dataframe has gaps in rows {gaps}. Leaves cannot be found. Fill "
            "gaps first."
        )
    lowest = []
    for i, row in df.iterrows():
        lowest.append((i, get_row_leaf(row)))
    return pd.DataFrame(lowest, columns=["idx", "value"]).set_index("idx").value


def add_root(df):
    """Adds a root column to a pandas dataframe"""
    df = df.copy()
    df.insert(0, "root", pd.Series(["root"] * len(df)))
    return df


def leaf_relations(row):
    """Handles the pairwise relations for a dataset row"""
    r = []
    for i in range(1, len(row)):
        if not pd.isna(row.iloc[-i]):
            r.append((row.iloc[-i], row.iloc[-i - 1]))
    return r


def check_for_gaps(df):
    """Checks for gaps in dataframe rows.
    If a row starts with values, changes to None, and then changes back to
    values, this function will detect it.
    For example [1, 2, None, None, 5] has a gap, while [1, 2, 3, None, None]
    does not.


    Args:
        df (pd.DataFrame): The dataframe to check for gaps.

    Returns:
        list: A list of indices where gaps occur. If no gaps are found,
              returns None.

    """
    A = df.isna().values * 1
    gaps = np.apply_along_axis(lambda x: np.abs(np.diff(x)).sum(), 1, A)
    if np.any(gaps > 1):
        gap_indices = np.where(gaps > 1)[0].tolist()
        return gap_indices
    return None


def fill_gaps(df, suffix="_fill"):
    """Fills gaps in a dataframe.
    If a row starts with values, changes to None, and then changes back to
    values, this function will fill the gap with the last value,
    appended with a suffix.
    For example ['a', 'b', None, 'c', None] will be filled to
    ['a', 'b', 'b_dummy', 'c', None].

    Args:
        df (pd.DataFrame): The dataframe to fill gaps in.

    Returns:
        pd.DataFrame: The dataframe with gaps filled.
    """
    df = df.copy()
    gaps = check_for_gaps(df)  # the gap indices
    A = df.isna().values * 1  # the dataframe as a binary matrix
    for i in gaps:  # iterate over rows with gaps
        r = df.iloc[i]  # row values
        nan_indices = np.where(A[i] == 1)[0]  # indices of NaN values
        last_nan_idx = (
            np.where(A[i] != 1)[0][-1] + 1
        )  # index of last non-NaN value, values after this will be NaN
        rnew = r.copy()  # copy the row
        for ni in nan_indices:  # iterate over NaN indices
            rnew.iloc[ni] = (
                str(rnew.iloc[ni - 1]) + suffix
            )  # fill NaN values with the previous value + "_fill"
        rnew.iloc[last_nan_idx:] = np.nan  # set values after the last
        # non-NaN value to NaN
        df.iloc[i] = rnew  # set the new row values
    return df


def _fill_gaps(df, suffix="_fill"):
    """Fills gaps in a dataframe."""
    return fill_gaps(df, suffix)


def table2rel(df):
    """Turns a pandas DataFrame into a relational list"""
    r = []
    for i, row in df.iterrows():
        r.append(leaf_relations(row))
    r = [item for sublist in r for item in sublist]
    rel = pd.DataFrame(r, columns=["names", "parents"]).drop_duplicates()
    return rel


def zip_series(names, parents):
    """Turns two series into a zipped list"""
    names = names.values.tolist()
    parents = parents.values.tolist()
    rel_list = list(zip(parents, names))
    return rel_list


# Bigtree functions
def rel2tree(rel, names_col="names", parents_col="parents"):
    """Turns a relational dataframe into a Bigtree tree"""
    names = rel[names_col].values.tolist()
    parents = rel[parents_col].values.tolist()
    rel_list = list(zip(parents, names))
    return bigtree.list_to_tree_by_relation(rel_list)


def check_duplicates_solvable(rel):
    """Checks whether the duplicates in the relation table are solvable."""
    duplicates = rel[rel["names"].duplicated(keep=False)]
    r = True
    for name in duplicates["names"].unique():
        parents = duplicates.set_index("names").loc[name]["parents"]
        n = len(parents)
        if n != 2:
            r = False
        if parents.isna().sum() != 1:
            r = False
    return r


def table2tree(df, names_col="names", parents_col="parents"):
    """Turns a pandas DataFrame into a Bigtree tree"""
    rel = table2rel(df)
    try:
        tree = rel2tree(rel, names_col=names_col, parents_col=parents_col)
    except ValueError as e:
        if rel["names"].is_unique:
            raise ValueError(
                f"{e}\n\nThe dataframe does not have a root node - you can "
                "automatically set one with set_root=True"
            )
        else:
            raise ValueError(
                f"{e}\n\nThe dataframe has duplicate name-parent pairs. "
                "Check nodes above and make each node has a single parent."
            )
    return tree


def get_parents(node: bigtree.Node, parents=None):
    """Get the parents of a node in a tree."""
    if parents is None:  # Initialize
        parents = []
    if node is not None:  # If node is not the root
        return get_parents(node.parent, parents + [node.name])
    else:
        return parents


def coarsen(labels, root, depth, return_map=False):
    """Coarsens a list of labels based on a depth in a tree"""
    new_map = {}
    for leaf in list(root.leaves):
        orig_label = leaf.name
        while leaf.depth > depth:
            leaf = leaf.parent

        new_map[orig_label] = leaf.name

    r = list(map(lambda x: new_map[x], labels))
    if return_map:
        return r, new_map
    return r


def prune_by_leaves(root, leaves):
    """Prunes a tree by a list of leaves that are left over"""
    paths = []
    for t in leaves:
        t = bigtree.find_name(root, t)
        if t:
            paths.append(t.path_name)
    return bigtree.list_to_tree(paths)


def simplify_tree(node):
    """Simplifies a tree so that every leaf has at least one sibling on the
    same level
    """
    leaf_list = [x for x in node.leaves]

    leaf_map = {}

    path_list = []
    for leaf in leaf_list:
        name = leaf.name
        try:
            while leaf.siblings == ():
                leaf = leaf.parent
            leaf_map[name] = leaf.name
            path_list.append(leaf.path_name)
        except AttributeError:
            warnings.warn(
                "The tree has a single branch. Simple tree is not displayed "
                "correctly"
            )
            return None, None

    return bigtree.list_to_tree(path_list), leaf_map


def LCA(root, node_a, node_b):
    """Returns the lowest common ancestor"""
    node_a = bigtree.find_path(root, node_a)
    node_b = bigtree.find_path(root, node_b)

    if (not node_a) or (not node_b):
        return None

    if node_a == node_b:
        return node_a

    # Find deeper node
    if node_a.depth == node_b.depth:
        deeper = node_a
        higher = node_b
    elif node_a.depth > node_b.depth:
        deeper = node_a
        higher = node_b
    else:
        deeper = node_b
        higher = node_a

    while deeper.depth != higher.depth:
        deeper = deeper.parent

    while deeper != higher:
        deeper = deeper.parent
        higher = higher.parent
    return deeper


def in_ancestors(root, node_a, ancestor):
    """Finds whether ancestor is the ancestor of node_a"""
    node = bigtree.find_name(root, node_a)
    node_ancestors = list(node.ancestors)
    ancestor = bigtree.find_name(root, ancestor)

    if ancestor.depth > node.depth:
        raise Exception("Ancestor is deeper than node")
    else:
        return ancestor in node_ancestors


def node_match(root, n1, n2):
    """Return strue if n1 and n2 match on any level"""
    try:
        a1 = in_ancestors(root, n1, n2)
    except Exception:
        a1 = in_ancestors(root, n2, n1)
    a2 = n1 == n2
    return a1 | a2


def remove_redundant_keys(d: dict) -> dict:
    """Removes key-value pairs from d where key==value"""
    d = d.copy()
    pop_keys = set()
    for k, v in d.items():
        if k == v:
            pop_keys.add(k)
    for k in pop_keys:
        d.pop(k)
    return d


class Tree:
    def __init__(self, df: pd.DataFrame, label_map=None, node_remapping=False):
        """Initializes a Tree object. Tree objects store the hierarchy.

        Args:
            df (pd.DataFrame): The hierarchy dataframe.
            label_map (dict, optional): Stores the mapping between labels and
                node names. Defaults to None.
            node_remapping (bool, optional): Flag to indicate whether node
                remapping is enabled. Defaults to False.

        Raises:
            ValueError: If a level column is named 'leaf' or 'simple'.

        Warnings:
            UserWarning: If the dataframe has gaps in rows and `fill_gaps` is
                not provided, the gaps will be filled with the last non-NaN
                value + '_fill'.
        """
        self.df = df  # The hierarchy dataframe

        if label_map is None:
            label_map = {}
        self.label_map = label_map
        self.node_remapping = node_remapping

        # Levels
        self.level_int = -1  # Current level as int. -1 is the leaf level

        # Set levels from dataframe columns and check that the format is ok
        assert columns_disjoint(df)
        self.levels = df.columns.tolist()
        self.levels_sortable = [f"{i:02d}_{s}" for i, s in enumerate(self.levels)]
        if "leaf" in self.levels:
            raise ValueError("A level column cannot be named 'leaf'")
        if "simple" in self.levels:
            raise ValueError("A level column cannot be named 'simple'")

        # Check for gaps in the dataframe
        gap_indices = check_for_gaps(df)
        if gap_indices:
            raise ValueError(
                f"Dataframe has gaps in rows {gap_indices}. "
                "Fill gaps first with tiers.fill_gaps(df), "
                "or set fill_gaps=True in tiers.Tree.from_dataframe"
            )

        # Create the tree and its simplified version
        self.root = table2tree(self.df)
        self.root_simple, self._leaf2simple_dict = simplify_tree(self.root)

        # Get the set of node names
        self.nodes = self.df.values.ravel().tolist()
        self.nodes = {x for x in self.nodes if pd.notna(x)}

        # Define mapping dicts
        self._level2sortable_dict = {
            k: v for k, v in zip(self.levels, self.levels_sortable)
        }
        self._levelstr2int_dict = {k: v for k, v in enumerate(self.levels)}
        self._levelint2str_dict = {k: v for v, k in self._levelstr2int_dict.items()}

        if label_map is not None:
            # Check that label map is ok
            if not node_remapping:
                self._check_label_map(self.label_map)

        # Update node attributes to match the label map
        self._set_node_attributes()
        # Flags
        self._node_as_label = False

        # Mapping cache
        self._map_cache = {}

    # Initialization
    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, node_remapping=False, set_root=False, fill_gaps=None
    ):
        """Initializes a tree from a dataframe, where the last column is the
        label column
        Initialization with this function handles dataframes that have
        labels as the last column, as well as filling gaps in the dataframe.

        Args:
            df (pd.DataFrame): The dataframe to initialize the tree from.
            node_remapping (bool, optional): Flag to indicate whether node
                remapping is enabled. Defaults to False.
            set_root (bool, optional): Flag to indicate whether a root node
                should be added. Defaults to False.
            fill_gaps: Whether to fill gaps in dataframe. Defaults to None.
                If False, raises an exception if gaps are found.
        """
        if set_root:
            df = add_root(df)
        # Separate label column and the actual hierarchy
        labels = df.iloc[:, -1]
        tree_df = df.iloc[:, :-1]

        # Check for gaps in the dataframe
        gap_indices = check_for_gaps(tree_df)
        if gap_indices:
            if fill_gaps:
                tree_df = _fill_gaps(tree_df)
            elif fill_gaps is None:
                tree_df = _fill_gaps(tree_df)
                warnings.warn(
                    f"Dataframe had gaps in rows {gap_indices}. "
                    "Filled with the last non-NaN value + '_fill'. "
                    "Set fill_gaps=True to fill gaps automatically "
                    "and suppress this warning"
                )
            else:
                raise Exception(
                    "Dataframe has gaps in rows. Fill gaps of set fill_gaps=True"
                )

        # Create the mapping from labels to leaves
        leaves = get_leaves(tree_df)
        z = zip(labels.values.tolist(), leaves.values.tolist())
        label_map = {label: value for label, value in z}
        return cls(df=tree_df, label_map=label_map, node_remapping=node_remapping)

    def _check_label_map(self, label_map):
        """Checks if any of the labels are in nodes and warns accordingly"""
        for k in label_map.keys():
            if k in self.nodes and label_map[k] != k:
                self._node_as_label = True
                warnings.warn(
                    f"label '{k}' is in the hierarchy as a higher-level node. "
                    f"It will map to '{label_map[k]}' by default, "
                    "unless nodes=True is set. "
                    "Remove the redundant row from the dataframe or set "
                    "`node_remapping=True` if you want to use node "
                    "names as labels and suppress this warning"
                )

    def _set_node_attributes(self):
        """Sets bigtree node attributes based on the label map"""
        for node_str in self.nodes:
            node = self.get_node(node_str)
            labels = self.labels(node_str)
            labels = [l for l in labels if l != node_str]
            if len(labels) > 0:
                node.set_attrs({"labels": labels})

    # Visualization
    def show(self, labels=False, **kwargs):
        """Shows the tree"""
        if labels:
            self.root.show(attr_list=["labels"], **kwargs)
        else:
            self.root.show(**kwargs)

    def subset(self, subset):
        """Returns a subset of the tree as a new tree"""
        if not isinstance(subset, list):
            raise ValueError("Subset must be a list of strings")

        rel = table2rel(self.df)  # Get the relational dataframe
        leaf_rel = rel[rel["names"].isin(subset)]  # Only the rows of the subset
        all_parents = []
        for x in leaf_rel["names"].values:  # Get all parents of the subset
            node = bigtree.find_name(self.root, x)  # Find the node for the leaf
            all_parents.append(get_parents(node))  # Get the parents of the leaf

        # Create a new dataframe with the parents
        dfnew = pd.DataFrame([x[::-1] for x in all_parents])
        if len(dfnew.columns) != len(self.levels):  # Add empty columns if necessary
            dfnew = pd.concat(
                [
                    dfnew,
                    pd.DataFrame(columns=range(len(self.levels) - len(dfnew.columns))),
                ],
                axis=1,
            )

        dfnew.columns = self.levels
        new_label_map = {k: v for k, v in self.label_map.items() if v in subset}

        return Tree(df=dfnew, label_map=new_label_map)

    def show_simple(self, labels=False):
        """Shows the simplified tree"""
        if labels:
            raise ValueError("Labels are not supported in simple tree")
        self.root_simple.show()

    # Levels
    @property
    def level(self):
        """Returns the level in sortable format"""
        if self.level_int == -1:
            return "leaf"
        return self.levels_sortable[self.level_int]

    def set_level(self, level: str):
        """Sets level by a string"""
        if level == "leaf":
            self.level_int = -1
            return
        try:
            self.level_int = self.level2int(level)
        except KeyError:
            raise KeyError(f"Invalid level. Levels are 'leaf' and : {self.levels}")

    def level2int(self, level: str):
        """Returns the integer level corresponding to the level string"""
        if level == "leaf":
            return -1
        if level == "simple":
            return -2
        return self._levelint2str_dict[level]

    # Nodes
    def get_node(self, node: str) -> bigtree.Node:
        """Returns a node by a name. Node must be in nodes, not in labels"""
        if node not in self.nodes:
            raise KeyError(
                f"Trying to get an nonexistent node {node}. Perhaps you are "
                "trying to find a label?"
            )
        return bigtree.find_name(self.root, node)

    def label2node(self, label: str) -> bigtree.Node:
        """Finds the node that corresponds to a label. 'label' must be in
        labels, not nodes
        """
        if label not in self.label_map.keys():
            raise KeyError(
                f"Trying to find a nonexistent label {label}. Perhaps you are "
                "trying to find a node?"
            )
        return self.get_node(self.label_map[label])

    def string2node(self, s: str) -> bigtree.Node:
        """Returns a node that best matches the string. First assumes the string
        is a label, and maps it to a node. If string is not in label list, uses
        it as a node name
        """
        if s in self.label_map.keys():
            return self.label2node(s)
        return self.get_node(s)  # Else in nodes

    @property
    def rel_dict(self):
        rel = table2rel(self.df)
        return {k: v for k, v in zip(rel["names"], rel["parents"])}

    @property
    def parent_map(self):
        return self.rel_dict

    @property
    def relation_table(self):
        return table2rel(self.df)

    @property
    def rel(self):
        return self.relation_table

    # Manipulation
    def merge(self, tree):
        """Merges two trees together"""
        df = (
            pd.concat((self.df, tree.df), axis=0)
            .reset_index(drop=True)
            .drop_duplicates()
        )
        new_label_map = self.label_map.copy()
        new_label_map.update(tree.label_map)
        return Tree(df=df, label_map=new_label_map)

    # Operations and search
    def lca(self, label_a, label_b, return_node=False, nodes=False):
        """Returns the lowest common ancestor of node_a and node_b. If
        nodes=False, maps labels first to nodes
        """
        if not nodes:
            label_a = self.string2node(label_a).name
            label_b = self.string2node(label_b).name
        node = LCA(self.root, label_a, label_b)
        if return_node:
            return node
        return node.name

    def in_ancestors(self, node_a, ancestor, nodes=False):
        """Finds whether ancestor is the ancestor of node_a. If nodes=False,
        maps labels first to nodes
        """
        if not nodes:
            node_a = self.string2node(node_a).name
        return in_ancestors(self.root, node_a, ancestor)

    def match(self, n1, n2, nodes=False):
        """Checks if n1 and n2 match on any level. If nodes=False, maps
        labels first to nodes
        """
        if not nodes:
            n1 = self.string2node(n1).name
            n2 = self.string2node(n2).name
        return node_match(self.root, n1, n2)

    def match_level(self, n1, n2, nodes=False):
        """Finds the match level of n1 and n2. If nodes=False, maps labels
        first to nodes
        """
        if not nodes:
            n1 = self.string2node(n1).name
            n2 = self.string2node(n2).name
        lca_node = self.lca(n1, n2, return_node=True)
        return self.levels_sortable[lca_node.depth - 1]

    def node_level(self, node):
        """Finds the level of a node. Node can be either a string or a
        node instance
        """
        if isinstance(node, bigtree.Node):
            return self.levels_sortable[node.depth - 1]
        return self.levels_sortable[self.get_node(node).depth - 1]

    def labels(self, nodes: Union[str, list]) -> list:
        """Returns the labels of the nodes

        Args:
            nodes (str | list): The node names

        Returns:
            list: a list of lists, where each list contains the labels of the nodes
        """
        if isinstance(nodes, str):
            return self.labels([nodes])[0]
        labels = []
        for node in nodes:
            node_labels = []
            for k, v in self.label_map.items():
                if v == node:
                    node_labels.append(k)
            labels.append(node_labels)
        return labels

    # Mapping values
    def map(self, labels, level: str = None, strict=False, nodes=False):
        """
        args:
            labels: labels as list or a string
            level: the level the labels will be mapped. By default uses the
                current level
            strict: If the label is lower than the level, will return None
            nodes: maps node strings instead of labels
        """
        if level:
            target_level = self.level2int(level)
        else:
            target_level = self.level_int

        if isinstance(labels, str):
            # Try to get the value first from the cache
            node_key = (labels, level, strict, nodes)
            if node_key in self._map_cache:
                return self._map_cache[node_key]

            # Otherwise we have to calculate it
            try:
                if target_level == -1:
                    if not nodes:
                        val = self.string2node(labels).name
                    else:
                        val = self.get_node(labels).name  # In case labels is a node
                    self._map_cache[node_key] = val
                    return val

                else:  # We need to traverse the tree
                    if not nodes:
                        node = self.string2node(labels)
                    else:
                        node = self.get_node(labels)

                    # If node is below the level and we want to return them
                    # as None
                    if (node.depth - 1 < target_level) and strict:
                        val = None
                        self._map_cache[node_key] = val
                        return val

                    # Otherwise
                    while node.depth - 1 > target_level:
                        node = node.parent
                    val = node.name
                    self._map_cache[node_key] = val
                    return val
            except Exception as e:
                raise e
        else:
            return [
                self.map(l, level=level, strict=strict, nodes=nodes) for l in labels
            ]

    def get_level(self, labels, nodes: bool = False, prefix: bool = True):
        """Maps a list of labels to the level it is on
        Args:
            labels: list of labels
            nodes (bool): if True, labels are node strings, not labels
            prefix (bool): if True, returns the level with the prefix,
                            e.g. '01_level', if False, returns 'level'
        Returns:
            str or list: the level of the label(s)
        """
        if isinstance(labels, str):  # If a single string is passed
            if not nodes:
                l = self.node_level(self.label2node(labels))
                if prefix:
                    return l
                return l[3:]
            l = self.node_level(labels)
            if prefix:
                return l
            return l[3:]
        return [self.get_level(l, nodes=nodes, prefix=prefix) for l in labels]

    def map_level(self, labels, nodes=False):
        warnings.warn("map_level is deprecated. Use get_level instead")

    def update_label_map(self, label_map: dict, node_remapping=None) -> "Tree":
        """Updates the label map. First asserts that all values are in nodes

        Args:
            label_map (dict): A dict that maps new values to nodes. Values must already
                be in nodes
        Returns:
            Tree: A new Tree object with the updated label_map
        Raises:
            ValueError: If a value in label_map is not in nodes

        Examples:
            >>> tree = tiers.Tree.from_dataframe(df)
            >>> tree = tree.update_label_map({"new_label": "old_node"})
        """
        for v in label_map.values():
            if v not in self.nodes:
                raise ValueError(f"Node '{v}' not in nodes")
        new_label_map = self.label_map.copy()
        new_label_map.update(label_map)
        return Tree(df=self.df, label_map=new_label_map, node_remapping=node_remapping)

    def longest_string_on_level(self, labels: Iterable, level: str):
        """Returns the longest string of list 'labels' when they are mapped to
        level 'level

        Args:
            labels (Iterable): list of labels. Only these are considered when
                calculating the longest list
            level (str): the labels are mapped to this level using tree.map()

        """
        return max(
            [len(self.map(l, level=level, strict=True) or "") for l in set(labels)]
        )

    def extend_label(self, label: str):
        """Returns a list of all levels associated with the 'label'

        Args:
            label (str): A label or a node name.
        Returns:
            list: A list of all parents associated with the parent
        """
        if not isinstance(label, str):
            raise ValueError("Label must be a string")
        parent_list = []
        for level in self.levels:
            nodename = self.map(label, level=level, strict=True)
            if nodename is None:
                nodename = ""
            parent_list.append(nodename)
        return parent_list + [label]

    def extend_labels(
        self,
        labels: Iterable[str],
        pad=False,
        return_string=False,
        levels=None,
        return_array=False,
    ):
        """Extends multiple labels. Optionally pads them for pretty printing

        Args:
            labels (Iterable): An iterable of labels or node names
            pad (bool): If True, each level will be padded by spaces to match
                the longest label on the level. Default True
            return_string (bool): If True, returns a single string, if False,
                returns a list of the strings. Default True
            levels (Iterable): List of levels that are only included
        Returns:
            (list | np.array): A list of the extended labels (or a numpy array
                    if return_array=True)
        """
        ext_arr = np.array([self.extend_label(l) for l in labels])

        if levels is not None:
            for l in levels:
                if l not in self.levels:
                    raise ValueError(
                        f"{l} not in tree levels. Available levels are {self.levels}"
                    )
            level_inds = [self.level2int(x) for x in levels]
            ext_arr = np.concatenate(
                (ext_arr[:, level_inds], ext_arr[:, -1].reshape(-1, 1)), axis=1
            )
        else:
            levels = self.levels

        if pad:
            pad_ns = [self.longest_string_on_level(labels, l) for l in levels]
            pad_ns += [max([len(s) for s in labels])]

            for i, p in enumerate(pad_ns):
                ext_arr[:, i] = np.array([s.rjust(p) for s in ext_arr[:, i]])
        if return_string:
            return [" - ".join(l) for l in ext_arr]
        if return_array:
            return ext_arr
        return [l for l in ext_arr]

    def extend_to_multiindex(self, labels: Iterable[str], levels=None) -> pd.MultiIndex:
        """Extends a list of labels into a pandas MultiIndex

        Args:
            labels (Iterable): An iterable of labels or node names
            levels (list, optional): A list of level names for the MultiIndex.
                    If not provided, the levels from the object will be used.

        Returns:
            pd.MultiIndex: A pandas MultiIndex of the labels
        """
        arr = self.extend_labels(labels, levels=levels, return_array=True)
        if levels is None:
            levels = self.levels
        idx = pd.MultiIndex.from_arrays(arr.T)
        idx = idx.rename(levels + ["label"])
        return idx

    def sort_labels(self, labels):
        """Sorts a list of labels according to their position in the hierarchy

        Args:
            tree (tiers.Tree): A tree object
            labels (Iterable): A list of labels
        Returns:
            list: A list of labels sorted according to the tree
        """

        cols = {}
        for level in self.levels:
            cols[level] = [self.map(x, level=level, strict=True) for x in labels]
        cols["label"] = labels
        taxa_table = pd.DataFrame(cols).fillna("")
        sort_table = taxa_table.sort_values(self.levels)
        return sort_table["label"].tolist()
