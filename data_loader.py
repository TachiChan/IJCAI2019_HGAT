import os
import numpy as np
import sklearn
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class ChunkSampler(Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class InteractionDataSet(Dataset):
    def __init__(self, file_dir, seed, shuffle, model, task):
        self.graphs = np.load(
            os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.float32)

        # self-loop trick, the input graphs should have no self-loop
        identity = np.identity(self.graphs.shape[1])
        self.graphs += identity
        self.graphs[self.graphs != 0] = 1.0
        if model == "gat":
            self.graphs = self.graphs.astype(np.dtype('B'))
        elif model == "gcn":
            # normalized graph laplacian for GCN: D^{-1/2}AD^{-1/2}
            for i in range(len(self.graphs)):
                graph = self.graphs[i]
                d_root_inv = 1. / np.sqrt(np.sum(graph, axis=1))
                graph = (graph.T * d_root_inv).T * d_root_inv
                self.graphs[i] = graph
        else:
            raise NotImplementedError
        print("graphs loaded!")

        if task == 'gender':
            labels = np.load(os.path.join(file_dir, "label_gender.npy")).astype(np.int64)
        elif task == 'age':
            labels = np.load(os.path.join(file_dir, "label_age.npy")).astype(np.int64)
        self.labels = np.argmax(labels, axis=1)
        print("labels loaded!")

        self.vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        print("vertex ids loaded!")

        word_features = np.load(
            os.path.join(file_dir, "word_feature_used_200.npy"))
        word_features = preprocessing.scale(word_features)
        self.word_features = torch.FloatTensor(word_features)
        print("global word features loaded!")

        interaction_item = np.load(
            os.path.join(file_dir, "interaction_item.npy")).astype(np.int64)
        self.interaction_item = torch.LongTensor(interaction_item)
        print("sample_neighs_user_item matrix loaded!")

        interaction_word = np.load(
            os.path.join(file_dir, "interaction_word.npy")).astype(np.int64)
        self.interaction_word = torch.LongTensor(interaction_word)
        print("sample_neighs_item_word matrix loaded!")

        if shuffle:
            self.graphs, \
                self.labels, self.vertices = sklearn.utils.shuffle(
                        self.graphs,
                        self.labels, self.vertices,
                        random_state=seed)

        self.N = self.graphs.shape[0]
        print(
            "%d ego networks loaded, each with size %d." % (
                self.N, self.graphs.shape[1]))

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.labels))
        self.class_weight = torch.FloatTensor(class_weight)

    def get_word_features(self):
        return self.word_features

    def get_feature_dimension(self):
        return self.word_features.shape[-1]

    def get_interaction_item(self):
        return self.interaction_item

    def get_interaction_word(self):
        return self.interaction_word

    def get_num_class(self):
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.vertices[idx]
