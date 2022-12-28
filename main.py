from typing import Union

import torch
from gensim import downloader
import numpy as np
import re

from matplotlib import pyplot as plt, animation
import time
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from gensim.models import Word2Vec, KeyedVectors
# import matplotlib.pyplot as plt
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm
from torch import nn, optim
from datasets import Dataset
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = ((y_pred_tag >= 1) & (y_test >= 1)).sum().float()
    correct_results_sum += ((y_pred_tag < 1) & (y_test < 1)).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


class NER:
    def __init__(self):
        self.is_fitted = False
        self.WORD_2_VEC_PATH = 'word2vec-google-news-300'
        self.GLOVE_PATH = 'glove-twitter-200'

    def f1_np(self, actual, predicted, label=1):
        """ A helper function to calculate f1-score for the given `label` """

        # F1 = 2 * (precision * recall) / (precision + recall)
        tp = np.sum((actual == np.full(len(actual), label)) & (predicted == np.full(len(actual), label)))
        fp = np.sum((actual != np.full(len(actual), label)) & (predicted == np.full(len(actual), label)))
        fn = np.sum((predicted != np.full(len(actual), label)) & (actual == np.full(len(actual), label)))

        f1 = (2 * tp) / (2 * tp + fn + fp)
        return f1


class BinaryModel(nn.Module):
    def __init__(self, word_representation_length, context_range):
        super(BinaryModel, self).__init__()
        # Number of input features is 12.
        l1_in = word_representation_length * (2 * context_range + 1)  # first linear layer
        self.output_layer = nn.Sequential(
            nn.Linear(l1_in, 1000),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1000, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(20, 1)
        )

    def forward(self, inputs):
        return self.output_layer(inputs)


class Predictor(nn.Module):
    def __init__(self, word_representation_length, context_range):
        super(Predictor, self).__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(word_representation_length * (2 * context_range + 1), 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 20),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(20, 1)
        )

    def forward(self, inputs):
        return self.output_layer(inputs)


class SimpleModel(NER):
    def __init__(self, file_path, rep_mode="w2v", predictor_type="svm", k_neighbour=3):
        """
        crate the simple model
        :param rep_mode: glove , train-glove , w2v or train-w2v
        :param predictor_type: svm or knn
        :param file_path: the path to the file containing the words with tags to be trained on
        :param k_neighbour: if predictor is knn the k to the model, else ignored
        """
        super().__init__()
        self.file_path = file_path
        self.rep_mode = rep_mode
        self.predictor_type = predictor_type
        self.k_neighbour = k_neighbour
        if rep_mode == "w2v":
            self.rep_dict = downloader.load(self.WORD_2_VEC_PATH)
        elif rep_mode == "train-w2v":
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences = f.readlines()
            sentences = [sen.strip().lower() for sen in sentences]
            sentences = [sen.split() for sen in sentences if sen]
            sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
            words = [entry[0] for entry in sentences]
            print("read file now training w2v")
            self.rep_dict = Word2Vec(sentences=words, vector_size=10, window=2, min_count=1, workers=4, epochs=100,
                                     seed=42).wv
            self.rep_dict.save(r'w2v_mode.rep')
        elif rep_mode == "glove":
            try:
                self.rep_dict = KeyedVectors.load(r'glove_mode.rep')
            except:
                self.rep_dict = downloader.load(self.GLOVE_PATH)
                self.rep_dict.save(r'glove_mode.rep')
        else:
            self.rep_dict = None

        # predictor mode
        if predictor_type == "knn":
            self.decider = KNeighborsClassifier(n_neighbors=k_neighbour)
        elif predictor_type == "svm":
            self.decider = svm.SVC()

    def get_representation(self, words: Union,
                           tags=None,
                           context_range=0):  ## TODO : take care of OOV (either by training or random choice)
        """
        represent words as vectors and remove tags for words not in dict if tags is supplied
        :param words: the words to represent
        :param tags: the appropriate tags to said words
        :return: (word-vector representation , tags to match th wrds)
        """
        representations = []
        valid_tags = []
        found_words = []
        if type(words) is not list:
            words = [words]
        for idx in range(len(words)):
            sub_sen_rep = []
            word_rep_mask = []
            vec_of_vecs = []
            for word_offset in range(-context_range, context_range + 1):
                if idx + word_offset < 0 or idx + word_offset >= len(words):
                    continue
                if words[idx + word_offset] not in self.rep_dict.key_to_index:
                    ## mark this and in the end fill this place
                    word_rep_mask.append(False)
                    sub_sen_rep.append(0)
                else:
                    word_rep_mask.append(True)
                    sub_sen_rep.append(np.asarray(self.rep_dict[words[idx + word_offset]]))
                    vec_of_vecs.append(np.asarray(self.rep_dict[words[idx + word_offset]]))
            average = np.mean(np.array(vec_of_vecs), axis=0)
            ## fill the blanks
            for cntxt_word in range(len(word_rep_mask)):
                if not word_rep_mask[cntxt_word]:
                    sub_sen_rep[cntxt_word] = average
            while len(sub_sen_rep) < 2 * context_range + 1:
                sub_sen_rep.append(average)
            if True in word_rep_mask:
                if tags is not None:
                    valid_tags.append(tags[idx])
                found_words.append(True)
                representations.append(np.concatenate(sub_sen_rep))
            else:
                found_words.append(False)
        if tags is None:
            return representations, found_words
        else:
            return representations, found_words, valid_tags

    def fit(self, context_range=0):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        sentences = [sen.strip().lower() for sen in sentences]
        sentences = [sen.split() for sen in sentences if sen]
        sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
        words = [entry[0] for entry in sentences]
        tags = [1 if entry[1] == 'o' else 0 for entry in sentences]
        word_vecs, _, valid_tags = self.get_representation(words=words, tags=tags, context_range=context_range)
        self.decider.fit(word_vecs, valid_tags)
        self.is_fitted = True

    def get_prediction(self, words_: Union = None, need_metric=False, new_file_path=None, context_range=0):
        """
        predict and print stats
        :param words_: (word,tag) pair
        :param need_metric: set to true if need to print the stats, else returns plain predictions
        :return: predictions over the words passed, id need_metric is false, does not compare
        """
        if not self.is_fitted:
            print("Your model was not fitted")
            return None
        if words_ is None:
            if new_file_path is None:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    sentences = f.readlines()
                sentences = [sen.strip().lower() for sen in sentences]
                sentences = [sen.split() for sen in sentences if sen]
                sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
                words = [entry[0] for entry in sentences]
                targets = [1 if entry[1] == 'o' else 0 for entry in sentences]
            else:
                with open(new_file_path, 'r', encoding='utf-8') as f:
                    sentences = f.readlines()
                sentences = [sen.strip().lower() for sen in sentences]
                sentences = [sen.split() for sen in sentences if sen]
                sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
                words = []
                targets = []
                for sen in sentences:
                    if len(sen) < 2:
                        continue
                    if sen[1] == 'o':
                        targets.append(1)
                    else:
                        targets.append(0)
                    words.append(sen[0])
        else:
            words = [w[0] for w in words_]
            try:
                targets = [w[1] for w in words_]
            except:
                targets = None
                print("no targets were found")

        word_vecs, found_words, _ = self.get_representation(words, targets, context_range)
        if need_metric:
            print(self.decider)
            preds = self.decider.predict(
                [word if found_words[idx] else word_vecs[0] for idx, word in enumerate(word_vecs)])
            predictions = []
            pred_indc = 0
            for i in range(len(found_words)):
                if found_words[i]:
                    predictions.append(preds[pred_indc])
                    pred_indc += 1
                else:
                    predictions.append(1)
            predictions = [predictions[idx] if found_words[idx] else 1 for idx in range(len(found_words))]
            accu = (np.asarray(targets) == np.asarray(predictions)).mean()
            f1 = self.f1_np(predictions, targets)
            print(
                f" with the rep model {self.rep_mode} and predictor {self.predictor_type} got accuracy of {accu} f1 score {f1}")
            # confusion_matrix_ = confusion_matrix(predictions, targets)
            # cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_)
            #
            # cm_display.plot()
            # plt.show()

            return predictions, (f1, accu)
        else:
            preds = self.decider.predict(word_vecs)
            pred_indc = 0
            predictions = []
            for i in range(len(found_words)):
                if found_words[i]:
                    predictions.append(preds[pred_indc])
                    pred_indc += 1
                else:
                    predictions.append(1)
            return predictions


class NNModel(SimpleModel):
    def __init__(self, file_path, context_range=1, word_representation_length=300, rep_mode="w2v"):
        super().__init__(file_path, rep_mode=rep_mode)
        self.context_range = context_range
        self.predictor = BinaryModel(word_representation_length, context_range)
        self.predictor = self.predictor.float()

    def fit(self, test_file=r'data/dev.tagged', num_epochs=10, to_tag=False):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        sentences = [sen.strip().lower() for sen in sentences]
        sentences = [sen.split() for sen in sentences if sen]
        sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
        train_words = [entry[0] for entry in sentences]
        train_targets = [1 if entry[1] == 'o' else 0 for entry in sentences]

        with open(test_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        sentences = [sen.strip().lower() for sen in sentences]
        sentences = [sen.split() for sen in sentences if sen]
        sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
        test_words = [entry[0] for entry in sentences]
        test_targets = [1 if entry[1] == 'o' else 0 for entry in sentences]
        self.predictor.to(device)

        optimizer = Adam(params=self.predictor.parameters(), lr=0.1)
        criterion = BCEWithLogitsLoss()
        best_val_err = 0.0
        best_f1 = 0.0
        words_rep, found_words, valid_tags = self.get_representation(words=train_words, tags=train_targets,
                                                                     context_range=self.context_range)
        f_1_lst = []
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            i = 0
            self.predictor.train()
            for idx in range(len(words_rep)):
                target = valid_tags[idx]
                optimizer.zero_grad()
                target_rep = torch.tensor(np.array([-10])) if target == 0 else torch.tensor(np.array([target]))
                word_rep = torch.tensor(words_rep[idx])
                word_rep = word_rep.float()
                # forward + backward + optimize
                outputs = self.predictor(word_rep)
                loss = criterion(outputs.type('torch.DoubleTensor'), target_rep.type('torch.DoubleTensor'))
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()

            self.predictor.eval()
            preds = []
            test_words_rep, test_found_words, test_valid_tags = self.get_representation(words=test_words,
                                                                                        tags=test_targets,
                                                                                        context_range=self.context_range)
            for idx in range(len(test_words_rep)):
                target = test_valid_tags[idx]
                optimizer.zero_grad()
                target_rep = torch.tensor(np.array([target]))
                word_rep = torch.tensor(test_words_rep[idx])
                word_rep = word_rep.float()
                # forward + backward + optimize
                outputs = self.predictor(word_rep)
                preds.append(1 if outputs.item() > 0.8 else 0)
            val_err = binary_acc(torch.tensor(np.array(test_valid_tags)), torch.tensor(np.array(preds)))
            f_1 = self.f1_np(np.array(test_valid_tags), np.array(preds))
            confusion_matrix_ = confusion_matrix(test_valid_tags, preds)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_)
            cm_display.plot()
            plt.show()
            if len(f_1_lst) >= 10:
                f_1_lst.pop(0)
            f_1_lst.append(f_1)
            f_1_avg = sum(f_1_lst) / len(f_1_lst)
            if epoch > 100 and all(i < f_1_avg for i in f_1_lst[5:]):
                print("reached the best validation error")
                print(f"the best validation error was {best_val_err} and best f1 was {best_f1}")
                return
            if best_f1 < f_1:
                best_val_err = val_err
                best_f1 = f_1
                print(f"best validation error is {best_val_err}, f1 score is {f_1}, saving model ...")
                torch.save(self.predictor.state_dict(), "winning_model")
            else:
                print(f"OoPss only got {val_err}, f1 score is {f_1}")

        print(f"the best validation error was {best_val_err} and best f1 was {best_f1}")

    def tag(self, range=0):
        self.predictor.load_state_dict(torch.load(r"winning_model"))
        self.predictor.eval()
        with open(r'data/test.untagged', 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        out_words = [sen if sen == '\n' else sen.strip() for sen in sentences]
        sentences = [sen.strip().lower() for sen in sentences]
        sentences = [sen.split() for sen in sentences if sen]
        sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
        train_words = [entry[0] for entry in sentences]
        words_rep, found_words = self.get_representation(words=train_words,
                                                         context_range=range)
        preds = []
        i = 0
        for idx, word in tqdm(enumerate(found_words)):
            if found_words[idx]:
                preds.append("O")
                continue
            word_rep = torch.tensor(words_rep[i])
            word_rep = word_rep.float()
            # forward + backward + optimize
            outputs = self.predictor(word_rep)
            preds.append("O" if outputs.item() > 0.8 else "N")
            i += 1
        i = 0
        with open('test.tagged', 'w') as f:
            for idx, word in enumerate(out_words):
                if out_words[idx] == '\n':
                    f.write(out_words[idx])
                    continue
                f.write(out_words[idx] + "\t" + preds[i] + "\n")
                i += 1


class ComplexModel(SimpleModel):
    def __init__(self, file_path, context_range=1, word_representation_length=300, rep_mode="w2v"):
        super().__init__(file_path, rep_mode=rep_mode)
        self.context_range = context_range
        self.predictor = Predictor(word_representation_length, context_range)
        self.predictor = self.predictor.float()

    def fit(self, test_file=r'data/dev.tagged', num_epochs=10, to_tag=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        sentences = [sen.strip().lower() for sen in sentences]
        sentences = [sen.split() for sen in sentences if sen]
        sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
        train_words = [entry[0] for entry in sentences]
        train_targets = [1 if entry[1] == 'o' else 0 for entry in sentences]

        with open(test_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        sentences = [sen.strip().lower() for sen in sentences]
        sentences = [sen.split() for sen in sentences if sen]
        sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
        test_words = [entry[0] for entry in sentences]
        test_targets = [1 if entry[1] == 'o' else 0 for entry in sentences]
        self.predictor.to(device)
        train_words_rep, train_found_words, train_valid_tags = self.get_representation(words=train_words, tags=train_targets,
                                                                     context_range=self.context_range)
        gittest_words_rep, test_found_words, test_valid_tags = self.get_representation(words=test_words,tags=test_targets,
                                                                                    context_range=self.context_range)
        optimizer = Adam(params=self.predictor.parameters(), lr=0.01)
        criterion = BCEWithLogitsLoss()
        best_val_err = 0.0
        best_f1 = 0.0

        fig_1 = plt.figure()
        fig_2 = plt.figure()
        ax1 = fig_1.add_subplot(1, 1, 1)
        ax2 = fig_2.add_subplot(1, 1, 1)
        val_errs = []
        f1_scores = []
        plt.show()
        for epoch in tqdm(range(num_epochs)):
            self.predictor.train()
            for idx, word in enumerate(train_words_rep):
                target = torch.tensor(np.array([1]) if train_valid_tags[idx] == 1 else np.array([1]))
                optimizer.zero_grad()
                word_tensor = (torch.tensor(train_words_rep[idx])).float()
                outputs = self.predictor(word_tensor)
                loss = criterion(outputs.type('torch.DoubleTensor'), target.type('torch.DoubleTensor'))
                loss.backward()
                optimizer.step()
                # print statistics                 running_loss += loss.item()

            self.predictor.eval()
            preds = []
            for idx, word in enumerate(test_words_rep):
                word_tensor = (torch.tensor(test_words_rep[idx])).float()
                outputs = self.predictor(word_tensor)
                preds.append(1 if outputs.item() > 0.8 else 0)
            val_err = binary_acc(torch.tensor(np.array(test_valid_tags)), torch.tensor(np.array(preds)))
            f_1 = self.f1_np(np.array(test_valid_tags), np.array(preds))
            val_errs.append(val_err)
            f1_scores.append(f_1)
            ax1.clear()
            ax1.plot(range(len(val_errs)), val_errs)
            ax2.clear()
            ax2.plot(range(len(f1_scores)), f1_scores)

            if best_f1 < f_1:
                best_val_err = val_err
                best_f1 = f_1
                print(f"best validation error is {best_val_err}, f1 score is {f_1}, saving model ...")
                torch.save(self.predictor.state_dict(), "winning_model")
            else:
                print(f"***** only got{f_1}")




if __name__ == "__main__":
    # best_k = -1
    # best_f1 = 0
    # best_context_range = -1
    # for k in tqdm(range(1, 10)):
    #     for context_range in range(2):
    #         model = SimpleModel(r'data/train.tagged', "glove", "knn", k) # K = 6 range = 0
    #         model.fit(context_range=context_range)
    #         res = model.get_prediction(need_metric=True, new_file_path=r'data/dev.tagged', context_range=context_range)
    #         if best_f1 < res[1][0]:
    #             best_f1 = res[1][0]
    #             best_k = k
    #             best_context_range = context_range
    # print(
    #     f"best k found for glove svm is {best_k} best context raneg is {best_context_range} with f1 score of {best_f1}")

    # best_k = -1
    # best_f1 = 0
    # best_context_range = -1
    # for k in tqdm(range(1, 10)):
    #     for context_range in range(2):
    #         model = SimpleModel(r'data/train.tagged', "w2v", "knn", k)  # K = 5
    #         model.fit(context_range=context_range)
    #         res = model.get_prediction(need_metric=True, new_file_path=r'data/dev.tagged', context_range=context_range)
    #         if best_f1 < res[1][0]:
    #             best_f1 = res[1][0]
    #             best_k = k
    # print(f"best k found for w2v svm is {best_k} best context raneg is {best_context_range} with f1 score of {best_f1}")

    # model_glove_svm = SimpleModel("glove", "svm", r'data/train.tagged')
    # model_glove_svm.fit()
    # model_glove_svm_pred = model_glove_svm.get_prediction(need_metric=True, new_file_path=r'data/dev.tagged')
    #
    # model_w2v_svm = SimpleModel("w2v", "svm", r'data/train.tagged')
    # model_w2v_svm.fit()
    # model_w2v_svm_pred = model_w2v_svm.get_prediction(need_metric=True, new_file_path=r'data/dev.tagged')
    #
    # model_trainw2v_svm = SimpleModel("train-w2v", "svm", r'data/train.tagged')
    # model_trainw2v_svm.fit()
    # model_trainw2v_svm_pred = model_trainw2v_svm.get_prediction(need_metric=True, new_file_path=r'data/dev.tagged')

    print("context range 2")
    model = ComplexModel(r'data/train.tagged', context_range=2)
    model.fit(num_epochs=300)
    print("glove")
    model = ComplexModel(r'data/train.tagged', context_range=2, word_representation_length=200, rep_mode="glove")
    model.fit(num_epochs=300)

    print("context range 1")
    model = ComplexModel(r'data/train.tagged', context_range=1)
    model.fit(num_epochs=300)
    print("glove")
    model = ComplexModel(r'data/train.tagged', context_range=1, word_representation_length=200, rep_mode="glove")
    model.fit(num_epochs=300)

    print("context range 0")
    model = ComplexModel(r'data/train.tagged', context_range=0)
    model.fit(num_epochs=300)
    print("glove")
    model = ComplexModel(r'data/train.tagged', context_range=0, word_representation_length=200, rep_mode="glove")
    model.fit(num_epochs=300)
