import os
import glob
import time
import argparse
import numpy as np
from sklearn import metrics

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from gcn import BatchGCN
from gat import BatchGAT
from data_loader import ChunkSampler
from data_loader import InteractionDataSet


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', type=str, default='gcn', help="Base model maybe gcn or gat.")
parser.add_argument(
    '--task', type=str, default='gender', help="Task maybe gender or age.")
parser.add_argument(
    '--no-cuda', action='store_true', default=False, help='Disable CUDA training.')
parser.add_argument(
    '--seed', type=int, default=42, help='Random seed.')
parser.add_argument(
    '--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument(
    '--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument(
    '--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument(
    '--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument(
    '--hidden-units', type=str, default="16,8", help="Hidden units in each hidden layer, splitted with comma.")
parser.add_argument(
    '--feature-size', type=str, default="200,200", help="feature size of item and user, splitted with comma.")
parser.add_argument(
    '--heads', type=str, default="1,1,1", help="Heads in each layer, splitted with comma.")
parser.add_argument(
    '--batch', type=int, default=64, help="Batch size.")
parser.add_argument(
    '--patience', type=int, default=10, help="Patience.")
parser.add_argument(
    '--instance-normalization', action='store_true', default=False, help="Enable instance normalization.")
parser.add_argument(
    '--shuffle', action='store_true', default=False, help="Shuffle dataset.")
parser.add_argument(
    '--data-dir', type=str, default='data', help="Data file directory.")
parser.add_argument(
    '--pkl-dir', type=str, default='00', help="Model file directory.")
parser.add_argument(
    '--train-ratio', type=float, default=75, help="Training ratio (0, 100).")
parser.add_argument(
    '--valid-ratio', type=float, default=12.5, help="Validation ratio (0, 100).")
parser.add_argument(
    '--class-weight-balanced', action='store_true', default=False, help="Adjust weights inversely proportional to class frequencies.")
parser.add_argument(
    '--use-user-feature', action='store_true', default=False, help="Whether to use users' structural features(include user node).")
parser.add_argument(
    '--use-item-feature', action='store_true', default=False, help="Whether to use items' structural features(include user and item nodes).")
parser.add_argument(
    '--use-word-feature', action='store_true', default=False, help="Whether to use words' structural features(include user, item and word nodes).")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

interaction_dataset = InteractionDataSet(
        args.data_dir, args.seed, args.shuffle, args.model, args.task)

N = len(interaction_dataset)
n_classes = interaction_dataset.get_num_class()
class_weight = interaction_dataset.get_class_weight() \
    if args.class_weight_balanced else torch.ones(n_classes)
print("class_weight: " + str(list(class_weight.numpy())))

feature_dim = interaction_dataset.get_feature_dimension()
item_dim, user_dim = [int(x) for x in args.feature_size.strip().split(",")]
n_units = [int(x) for x in args.hidden_units.strip().split(",")] + [n_classes]
print("feature dimension={}".format(feature_dim))
print("number of classes={}".format(n_classes))

train_start,  valid_start, test_start = 0, \
    int(N * args.train_ratio / 100), \
    int(N * (args.train_ratio + args.valid_ratio) / 100)
train_loader = DataLoader(
    interaction_dataset, batch_size=args.batch,
    sampler=ChunkSampler(valid_start - train_start, 0))
valid_loader = DataLoader(
    interaction_dataset, batch_size=args.batch,
    sampler=ChunkSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(
    interaction_dataset, batch_size=args.batch,
    sampler=ChunkSampler(N - test_start, test_start))

interaction_item = interaction_dataset.get_interaction_item()
interaction_word = interaction_dataset.get_interaction_word()

if args.cuda:
    interaction_item = interaction_item.cuda()
    interaction_word = interaction_word.cuda()

# Model and optimizer
if args.model == "gcn":
    model = BatchGCN(
        word_feature=interaction_dataset.get_word_features(),
        interaction_item=interaction_item,
        interaction_word=interaction_word,
        use_user_feature=args.use_user_feature,
        use_item_feature=args.use_item_feature,
        use_word_feature=args.use_word_feature,
        n_units=n_units,
        item_dim=item_dim, user_dim=user_dim,
        dropout=args.dropout,
        instance_normalization=args.instance_normalization)
elif args.model == "gat":
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = BatchGAT(
        word_feature=interaction_dataset.get_word_features(),
        interaction_item=interaction_item,
        interaction_word=interaction_word,
        use_user_feature=args.use_user_feature,
        use_item_feature=args.use_item_feature,
        use_word_feature=args.use_word_feature,
        n_units=n_units, n_heads=n_heads,
        item_dim=item_dim, user_dim=user_dim,
        dropout=args.dropout,
        instance_normalization=args.instance_normalization)
else:
    raise NotImplementedError

if args.cuda:
    model.cuda()
    class_weight = class_weight.cuda()

params = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]
optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)


def classification_metrics(y_true, y_pred):
        acc = float(metrics.accuracy_score(y_true, y_pred))
        # micro_f1 = float(metrics.f1_score(y_true, y_pred, average="micro"))
        macro_f1 = float(metrics.f1_score(y_true, y_pred, average="macro"))
        # return acc, micro_f1, macro_f1
        return acc, macro_f1


def train(epoch, train_loader, valid_loader):
    t = time.time()
    model.train()
    loss_train = 0.
    total_train = 0.
    y_true_train, y_pred_train, y_score_train = [], [], []
    for _, batch in enumerate(train_loader):
        graph, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        optimizer.zero_grad()
        output, _ = model(vertices, graph)
        if args.model == "gcn" or args.model == "gat":
            output = output[:, -1, :]
        loss_train_batch = F.nll_loss(output, labels, class_weight)
        loss_train += bs * loss_train_batch.item()

        y_true_train += labels.data.tolist()
        y_pred_train += output.max(1)[1].data.tolist()
        y_score_train += output[:, 1].data.tolist()
        total_train += bs
        loss_train_batch.backward()
        optimizer.step()

    model.eval()
    loss_val = 0.
    total_val = 0.
    y_true_val, y_pred_val, y_score_val = [], [], []
    for _, batch in enumerate(valid_loader):
        graph, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        output, _ = model(vertices, graph)
        output = output[:, -1, :]
        loss_val_batch = F.nll_loss(output, labels, class_weight)
        loss_val += bs * loss_val_batch.item()

        y_true_val += labels.data.tolist()
        y_pred_val += output.max(1)[1].data.tolist()
        y_score_val += output[:, 1].data.tolist()
        total_val += bs

    # acc_train, macro_f1_train = classification_metrics(y_true_train, y_pred_train)
    acc_val, macro_f1_val = classification_metrics(y_true_val, y_pred_val)

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train / total_train),
        #   'acc_train: {:.4f}'.format(acc_train),
        #   'macro_f1_train: {:.4f}'.format(macro_f1_train),
          '||',
          'loss_val: {:.4f}'.format(loss_val / total_val),
          '||',
          'acc_val: {:.4f}'.format(acc_val),
          'macro_f1_val: {:.4f}'.format(macro_f1_val),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val / total_val


def compute_test(test_loader):
    model.eval()
    loss_test = 0.
    total_test = 0.
    y_true, y_pred, y_score = [], [], []
    emb_test = []
    for _, batch in enumerate(test_loader):
        graph, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        output, test_emb = model(vertices, graph)
        emb_test += test_emb.data.tolist()

        output = output[:, -1, :]
        loss_test_batch = F.nll_loss(output, labels, class_weight)
        loss_test += bs * loss_test_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total_test += bs

    acc_test, macro_f1_test = classification_metrics(y_true, y_pred)

    # for t-SNE
    np.save(os.path.join(args.pkl_dir, 'emb_test.npy'), np.array(emb_test))
    np.save(os.path.join(args.pkl_dir, 'label_test.npy'), np.array(y_true))

    print("Test set results:",
          "loss= {:.4f}".format(loss_test / total_test),
          "accuracy= {:.4f}".format(acc_test),
          'macro_f1=: {:.4f}'.format(macro_f1_test))
    
    # print("confusion matrix: ", metrics.confusion_matrix(y_true, y_pred))


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch, train_loader, valid_loader))
    torch.save(model.state_dict(), os.path.join(args.pkl_dir, '{}.pkl'.format(epoch)))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob(os.path.join(args.pkl_dir, '*.pkl'))
    for file in files:
        epoch_nb = int(file.split('/')[-1].split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob(os.path.join(args.pkl_dir, '*.pkl'))
for file in files:
    epoch_nb = int(file.split('/')[-1].split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(os.path.join(args.pkl_dir, '{}.pkl'.format(best_epoch))))

# Testing
compute_test(test_loader)
