import pytest
import tiers
import numpy as np

@pytest.mark.usefixtures("df_taxa_table_no_labels")
def test_get_leaves(df_taxa_table_no_labels):
    leaves = tiers.get_leaves(df_taxa_table_no_labels)

    assert (leaves == np.array(
        ['Asellus aquaticus', 'Caenis horaria', 'Caenis luctuosa',
       'Caenis rivulorum', 'Centroptilum luteolum', 'Cloeon dipterum',
       'Cyrnus trimaculatus', 'Ecnomus tenellus', 'Ephemera vulgata',
       'Erpobdella octoculata', 'Helobdella stagnalis',
       'Heptagenia dalecarlica', 'Kageronia fuscogrisea',
       'Lepidostoma hirtum', 'Mystacides azureus',
       'Oulimnius tuberculatus larva', 'Oulimnius tuberculatus adult',
       'Oulimnius tuberculatus larva', 'Polycentropus flavomaculatus',
       'Psychomyia pusilla', 'Radix balthica', 'Spirosperma ferox',
       'Stylaria lacustris', 'Tinodes waeneri']
       )).all()

@pytest.mark.usefixtures("df_all_gaps")
def test_get_leaves(df_all_gaps):
    with pytest.raises(ValueError) as excinfo:
        tiers.get_leaves(df_all_gaps)
    assert "Dataframe has gaps in rows [1, 2, 3, 4, 5]. Leaves cannot be found. Fill gaps first." in str(excinfo.value)