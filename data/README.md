# Data Description

| Data | Size | Element Description |
| ---- | ---- | ---- |
| word_feature.npy | num_word * dim_word | Word embedding pre-trained by FastText. |
| adjacency_matrix.npy | num_user *  num_sample_user * num_sample_user| The adjacency matrix of User-User mini subgraph, which consists of 0 or 1, and the subgraph has no self-loop. |
| vertex_id.npy | num_user * num_sample_user | The vertex ids of User-User mini subgraph, we set the last vertex in each subgraph as the ego user node here. |
| interaction_item.npy | num_user * num_sample_item | Item-User mini subgraph, which consists of sampled item ids. |
| interaction_word.npy | num_item * num_sample_word | Attribute-Item mini subgraph, which consists of sampled attribute ids. |
| label_gender.npy | num_user * dim_gender | One-hot matrix. |
| label_age.npy | num_user * dim_age | One-hot matrix. |

The dataset used in our paper can be downloaded here: https://github.com/guyulongcs/IJCAI2019_HGAT.
