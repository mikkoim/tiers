# tiers
A hierarchical label handling library for Python

# Installation

You can install tiers using pip:

```bash
pip install git+https://github.com/mikkoim/tiers.git
```

# Examples

## Creating a tiers Tree from a Pandas DataFrame:
```python
import pandas as pd
from tiers import Tree

# Sample DataFrame with hierarchical data
data = {
    "Level1": ["A", "A", "B", "B", "B"],
    "Level2": ["A1", "A2", None, "B2", "B3"],
    "Level3": ["A1a", "A2b", None, None, "B3c"],
    "Label": ["Label1", "Label2", "Label3", "Label4", "Label5"]
}
df = pd.DataFrame(data)

# Create a Tree object from the DataFrame
tree = Tree.from_dataframe(df, set_root=True)

# Show the tree
tree.show()
# root
# ├── A
# │   ├── A1
# │   │   └── A1a
# │   └── A2
# │       └── A2b
# └── B
#     ├── B2
#     └── B3
#         └── B3c
```

## Mapping labels to nodes at a specific level

```python
# Map labels to nodes at a specific level
mapped_nodes = tree.map(["Label1", "Label2", "Label3", "Label4", "Label5"],
                        level="Level2")
print(mapped_nodes)  # Output: ['A1', 'A2', 'B', 'B2', 'B3']

# Mapping can be also done stricty on specific level
level3_labels = tree.map(["Label5", "Label4", "Label3"],
                        level="Level3",
                        strict=True)
print(level3_labels) # Output: ['B3c', None, None]
# 
```

## Setting the tree to a level and mapping labels to it
```python
tree.set_level("Level1")
mapped_nodes = tree.map(["Label1", "Label2", "Label3", "Label4", "Label5"])
print(mapped_nodes) # Output: ['A', 'A', 'B', 'B', 'B']
```

## Finding the Lowest Common Ancestor for two nodes:

```python
# Find the Lowest Common Ancestor (LCA) of two nodes
print(tree.lca("Label2", "Label4")) # Output: 'root'
print(tree.lca("Label1", "Label2")) # Output: 'A'
```

See the [demo notebook](docs/demo.ipynb) for more details.


# Acknowledgements

This library uses the excellent [`bigtree`](https://github.com/kayjan/bigtree) library, which provides support for handling tree structures.

# Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request. The library is still under heavy development and breaking changes can be introduced.
