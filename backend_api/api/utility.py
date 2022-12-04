import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0007
NUM_CLASSES = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FILE = '/home/shreya/major-backend/backend_api/api/model.pth'
scaler = MinMaxScaler()


def get_correlation(df):
    relation = df.corr()
    relation['label_no'].sort_values().drop('label_no').plot(kind='bar')
    relation['label_no'].sort_values().drop('label_no')
    df.rename(columns={'Unnamed: 0': 'samples'}, inplace=True)
    significant_genes = []
    df.loc[:, [0]]

    significant_genes = ['samples']

    for col in relation.columns[1:]:
        if relation.iloc[-1][col] < -0.4 or relation.iloc[-1][col] > 0.4:
            significant_genes.append(col)

    sub_df = df
    sub_df = sub_df.loc[:, significant_genes]
    sub_df.to_csv('Significant Data.csv')


def relief():
    df = pd.read_csv('Significant Data.csv')
    X = df.iloc[:, 2: -1]

    values = {1: 'COAD', 2: "PRAD", 3: "LUAD", 4: "BRCA", 5: "KIRC"}
    l = []
    for i in range(len(df)):
        l.append(values[df["label_no"][i]])
    Y = l

    X2 = X.to_numpy()

    query_cols = X.columns
    cols_index = [X.columns.get_loc(col) for col in query_cols]
    map = {}
    i = 0
    for k in X.columns:
        map[i] = k
        i += 1

    test = SelectKBest(score_func=f_classif, k=1500)
    fit = test.fit(X2, Y)
    features = fit.transform(X2)

    s = {}
    for i in range(len(fit.scores_)):
        s[map[i]] = fit.scores_[i]

    sorted_scores = sorted(s.items(), key=lambda x: x[1], reverse=True)

    significant_l = []
    for i in range(1500):
        significant_l.append(sorted_scores[i][0])

    sub_df1 = df
    sub_df1 = sub_df1.loc[:, significant_l]
    sub_df1['labels'] = Y
    sub_df1.to_csv('Significant_Feature_Selection.csv')


def lda():
    df = pd.read_csv('Significant_Feature_Selection.csv')
    X = df.iloc[:, 1: -1]
    Y = df['labels']

    lda = LinearDiscriminantAnalysis(n_components=4)
    X_lda = lda.fit(X, Y).transform(X)
    lda_df = pd.DataFrame(
        X_lda, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    lda_df['labels'] = Y
    lda_df.to_csv('LDA.csv')

    return lda_df


def preprocess():
    df = pd.read_csv('data.csv')
    labels = pd.read_csv('labels.csv')
    df["labels"] = labels["Class"]

    label_values = {'COAD': 1, "PRAD": 2, "LUAD": 3, "BRCA": 4, "KIRC": 5}
    label_number = []

    for i in range(len(df)):
        label_number.append(label_values[df["labels"][i]])

    df["label_no"] = label_number
    df = df.drop('labels', axis=1)

    return df


def get_class_distribution(obj):
    count_dict = defaultdict(int)

    for i in obj:
        if i == 0:
            count_dict['COAD'] += 1
        elif i == 1:
            count_dict['PRAD'] += 1
        elif i == 2:
            count_dict['LUAD'] += 1
        elif i == 3:
            count_dict['BRCA'] += 1
        elif i == 4:
            count_dict['KIRC'] += 1

    return count_dict


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def GetModel(df):
    arr_x = df.to_numpy()
    arr_y = df['labels'].to_list()

    label_values = {'COAD': 0, "PRAD": 1, "LUAD": 2, "BRCA": 3, "KIRC": 4}

    idx2class = {v: k for k, v in label_values.items()}

    df['labels'].replace(label_values, inplace=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    NUM_FEATURES = len(X.columns)
    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.6, stratify=y, random_state=69)

    # Split train into train-val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    # X_test, y_test = np.array(X_test), np.array(y_test)

    train_dataset = ClassifierDataset(torch.from_numpy(
        X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(
        X_val).float(), torch.from_numpy(y_val).long())
    # test_dataset = ClassifierDataset(torch.from_numpy(
    #     X_test).float(), torch.from_numpy(y_test).long())

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)

    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MulticlassClassification(
        num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("Begin training.")
    for e in tqdm(range(1, EPOCHS+1)):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()

        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(
                device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(
                    device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
    return model


def predict(fname):
    print(fname)
    X = pd.read_csv(fname)
    print(X)
    X = X.iloc[:, 2:]

    Y = np.random.randint(0, 5, len(X))
    print(Y)
    df1 = X
    # scaler = MinMaxScaler()
    X = scaler.transform(X)
    print(X)
    X, Y = np.array(X), np.array(Y)

    predict_dataset = ClassifierDataset(
        torch.from_numpy(X).float(), torch.from_numpy(Y).long())
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=1)

    y_pred_list = []
    with torch.no_grad():
        print("mode testing startedddddddddddddddddddddddddddddddddd")
        model = MulticlassClassification(*args, **kwargs)
        model.load_state_dict(torch.load(FILE))
        model.eval()
        print("fjhwaieuhfvljkbearihwiafhbidlsbvwaifyrufedck")
        for X_batch, _ in predict_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print(y_pred_list)
    label_number = {0: 'COAD', 1: "PRAD", 2: "LUAD", 3: "BRCA", 4: "KIRC"}
    label_values = []
    for i in range(len(y_pred_list)):
        label_values.append(label_number[y_pred_list[i]])

    df1['result'] = label_values
    result_file = fname + '_results.csv'
    df1.to_csv(result_file)

    return result_file
    # X = pd.read_csv(fname)
    # Y = np.random.randint(0, 5, len(X))

    # # scaler = MinMaxScaler()
    # # X_train = scaler.fit_transform(X_train)
    # # X = scaler.transform(X)
    # X, Y = np.array(X), np.array(Y)

    # predict_dataset = ClassifierDataset(
    #     torch.from_numpy(X).float(), torch.from_numpy(Y).long())
    # predict_loader = DataLoader(dataset=predict_dataset, batch_size=1)

    # y_pred_list = []
    # with torch.no_grad():
    #     model = MulticlassClassification(*args, **kwargs)
    #     model.load_state_dict(torch.load(FILE))
    #     model.eval()

    #     for X_batch, _ in predict_loader:
    #         X_batch = X_batch.to(device)
    #         y_test_pred = model(X_batch)
    #         _, y_pred_tags = torch.max(y_test_pred, dim=1)
    #         y_pred_list.append(y_pred_tags.cpu().numpy())
    # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    # X['result'] = y_pred_list
    # result_file = fname + '_result.csv'
    # X.to_csv(result_file)

    # return result_file


# df = preprocess()
# get_correlation(df)
# relief()
#processed_df = lda()


# processed_df = pd.read_csv('data.csv')
# processed_df = processed_df.iloc[:, 2:]
# labels = pd.read_csv('labels.csv')
# processed_df['labels'] = labels['Class']


#a = GetModel(processed_df)
# print(a.state_dict())
#torch.save(a.state_dict(), FILE)
