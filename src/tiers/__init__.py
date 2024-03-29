"""
Tiers - a hierarchical label handling library
"""
import bigtree
import pandas as pd
import warnings

def columns_disjoint(df):
    """Checks if dataframe columns are disjoint from each other"""
    s = set()
    for col in df.columns:
        s2 = set(df[col].dropna().unique().tolist())
        if s.isdisjoint(s2):
            s = s.union(s2)
        else:
            raise ValueError(f"Columns are not disjoint. {s.intersection(s2)} exist multiple times")
    return True

def get_row_leaf(row):
    "Returns the leftmost not-NaN value of a pandas row"
    for i in range(len(row)):
        if pd.isna(row[i]):
            return row[i-1]
    return row[i]

def get_leaves(df) -> pd.Series:
    """Find the leaves (leftmost not-NaN value of each row) for a dataframe"""
    lowest = []
    for i, row in df.iterrows():
        lowest.append((i, get_row_leaf(row)))
    return pd.DataFrame(lowest, columns=["idx", "value"]).set_index("idx").value
    
def add_root(df):
    """Adds a root column to a pandas dataframe"""
    df = df.copy()
    df.insert(0, "root", pd.Series(["root"]*len(df)))
    return df

def leaf_relations(row):
    """Handles the pairwise relations for a dataset row"""
    r = []
    for i in range(1,len(row)):
        if not pd.isna(row.iloc[-i]):
            r.append((row.iloc[-i], row.iloc[-i-1]))
    return r
    
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

def table2tree(df, names_col="names", parents_col="parents"):
    """Turns a pandas DataFrame into a Bigtree tree"""
    rel = table2rel(df)
    try:
        tree = rel2tree(rel, names_col=names_col, parents_col=parents_col)
    except ValueError as e:
        raise ValueError(f"{e} - The dataframe does not have a root node - you can automatically set one with set_root=True")

    return tree

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
    """Simplifies a tree so that every leaf has at least one sibling on the same level"""
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
            warnings.warn("The tree has a single branch. Simple tree is not displayed correctly")
            return None, None

    return bigtree.list_to_tree(path_list), leaf_map

def LCA(root, node_a, node_b):
    """Returns the lowest common ancestor """
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
    for k,v in d.items():
        if k==v:
            pop_keys.add(k)
    for k in pop_keys:
        d.pop(k)
    return d


class Tree():
    def __init__(self,
                 df: pd.DataFrame,
                 label_map = None,
                 node_remapping=False,
                 set_root=False):
        """Initializes a Tree object. Tree objects store the hierarchy"""
        self.df = df # The hierarchy dataframe

        if label_map is None:
            label_map = {}
        self.label_map = label_map # Stores the mapping between labels and node names

        # Levels
        self.level_int = -1 # Current level as int. -1 is the leaf level

        # Set levels from dataframe columns and check that the format is ok
        assert columns_disjoint(df)
        self.levels = df.columns.tolist()
        self.levels_sortable = [f"{i:02d}_{s}" for i,s in enumerate(self.levels)]
        if "leaf" in self.levels: raise ValueError("A level column cannot be named 'leaf'")
        if "simple" in self.levels: raise ValueError("A level column cannot be named 'simple'")

        # Create the tree and its simplified version
        self.root = table2tree(df)
        self.root_simple, self._leaf2simple_dict = simplify_tree(self.root)

        # Get the set of node names
        self.nodes = set(df.values.ravel().tolist())

        # Define mapping dicts
        self._level2sortable_dict = {k:v for k,v in zip(self.levels, self.levels_sortable)}
        self._levelstr2int_dict = {k:v for k,v in enumerate(self.levels)}
        self._levelint2str_dict = {k:v for v,k in self._levelstr2int_dict.items()}

        if label_map is not None:
            # Check that label map is ok
            if not node_remapping:
                self._check_label_map(self.label_map)
        
        # Flags
        self._node_as_label = False

    # Initialization
    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       node_remapping=False,
                       set_root=False):
        """Initializes a tree from a dataframe, where the last column is the label column"""
        if set_root:
            df = add_root(df)
        # Separate label column and the actual hierarchy
        labels = df.iloc[:,-1]
        tree_df = df.iloc[:,:-1]

        # Create the mapping from labels to leaves
        leaves = get_leaves(tree_df)
        z = zip(labels.values.tolist(), leaves.values.tolist())
        label_map = {label: value for label,value in z}
        return cls(df=tree_df,
                   label_map=label_map,
                   node_remapping=node_remapping,
                   set_root=set_root)
    
    def _check_label_map(self, label_map):
        """Checks if any of the labels are in nodes and warns accordingly"""
        for k in label_map.keys():
            if k in self.nodes and label_map[k] != k:
                self._node_as_label = True
                warnings.warn(f"label '{k}' is in the hierarchy as a node. Set `node_remapping=True` if you want to use node names as labels and suppress this warning")

    # Visualization
    def show(self, **kwargs):
        """Shows the tree"""
        self.root.show(**kwargs)

    def show_simple(self):
        """Shows the simplified tree"""
        self.root_simple.show()

    # Levels
    @property
    def level(self):
        """Returns the level in sortable format"""
        if self.level_int == -1:
            return "leaf"
        return self.levels_sortable[self.level_int]

    def set_level(self, level:str):
        """Sets level by a string"""
        if level == "leaf":
            self.level_int = -1
            return 
        try:
            self.level_int = self.level2int(level)
        except KeyError:
            raise KeyError(f"Invalid level. Levels are 'leaf' and : {self.levels}")
    
    def level2int(self, level:str):
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
            raise KeyError(f"Trying to get an nonexistent node {node}. Perhaps you are trying to find a label?")
        return bigtree.find_name(self.root, node)

    def label2node(self, label: str) -> bigtree.Node:
        """Finds the node that corresponds to a label. 'label' must be in labels, not nodes"""
        if label not in self.label_map.keys():
            raise KeyError(f"Trying to find a nonexistent label {label}. Perhaps you are trying to find a node?")
        return self.get_node(self.label_map[label])

    def string2node(self, s: str) -> bigtree.Node:
        """Returns a node that best matches the string. First assumes the string
        is a label, and maps it to a node. If string is not in label list, uses
        it as a node name
        """
        if s in self.label_map.keys():
            return self.label2node(s)
        return self.get_node(s) # Else in nodes
        
    # Manipulation
    def merge(self, tree):
        """Merges two trees together"""
        df = pd.concat((self.df, tree.df), axis=0).reset_index(drop=True).drop_duplicates()
        new_label_map = self.label_map.copy()
        new_label_map.update(tree.label_map)
        return Tree(df = df,
                    label_map=new_label_map)
    
    # Operations and search
    def lca(self, label_a, label_b, return_node=False, nodes=False):
        """Returns the lowest common ancestor of node_a and node_b. If nodes=False, maps labels first to nodes"""
        if not nodes:
            label_a = self.string2node(label_a).name
            label_b = self.string2node(label_b).name
        node = LCA(self.root, label_a, label_b)
        if return_node:
            return node
        return node.name

    def in_ancestors(self, node_a, ancestor, nodes=False):
        """Finds whether ancestor is the ancestor of node_a. If nodes=False, maps labels first to nodes"""
        if not nodes:
            node_a = self.string2node(node_a).name
        return in_ancestors(self.root, node_a, ancestor)

    def match(self, n1, n2, nodes=False):
        """Checks if n1 and n2 match on any level. If nodes=False, maps labels first to nodes"""
        if not nodes:
            n1 = self.string2node(n1).name
            n2 = self.string2node(n2).name
        return node_match(self.root, n1, n2)
    
    def match_level(self, n1, n2, nodes=False):
        """Finds the match level of n1 and n2. If nodes=False, maps labels first to nodes"""
        if not nodes:
            n1 = self.string2node(n1).name
            n2 = self.string2node(n2).name
        lca_node = self.lca(n1, n2, return_node=True)
        return self.levels_sortable[lca_node.depth-1]
    
    def node_level(self, node):
        """Finds the level of a node. Node can be either a string or a node instance"""
        if isinstance(node, bigtree.Node):
            return self.levels_sortable[node.depth-1]
        return self.levels_sortable[self.get_node(node).depth-1]


    def map(self, labels, level:str=None, strict=False, nodes=False):
        """
        args:
            labels: labels as list or a string
            level: the level the labels will be mapped. By default uses the current level
            strict: If the label is lower than the level, will return None
            nodes: maps node strings instead of labels
        """
        if level:
            target_level = self.level2int(level)
        else:
            target_level = self.level_int

        if isinstance(labels, str):
            try:
                if target_level == -1:
                    if not nodes:
                        return self.string2node(labels).name
                    return self.get_node(labels).name # In case labels is a node string
                
                else: # We need to traverse the tree 
                    if not nodes:
                        node = self.string2node(labels)
                    else:
                        node = self.get_node(labels)

                    # If node is below the level and we want to return them as None
                    if (node.depth-1 < target_level) and strict:
                        return None
                    
                    # Otherwise
                    while node.depth-1 > target_level:
                        node = node.parent
                    return node.name
            except Exception as e:
                raise e
        else:
            return [self.map(l, level=level, strict=strict, nodes=nodes) for l in labels]


    def map_level(self, labels, nodes=False):
        """Maps a list of labels to the level it is on"""
        if isinstance(labels, str):
            if not nodes:
                return self.node_level(self.label2node(labels))
            return self.node_level(labels)
        return [self.map_level(l, nodes=nodes) for l in labels]
