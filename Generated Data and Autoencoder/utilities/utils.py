import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Args:
        cm: confusion matrix to be plotted. Most likely it will be the output of sklearn's confusion_matrix function.
        classes: list of classes that correspond to the columns and rows of the confusion_matrix
        normalize: bool, whether to normalize or not the input confusion matrix
    """
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)
                      [:, np.newaxis], decimals=6)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    plt.imshow(cm_n, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plothistory(h, metric='accuracy', validation=True):
    """
    Args:
            h: a tensorflow history object
            metric: string, metric to be displayed
            validation: bool, whether to plot results also for the validation data
    """
    if metric == 'accuracy':
        plt.title('accuracy')
    else:
        plt.title(metric)

    if metric == 'mse':
        plt.yscale('log')

    if validation:
        plt.plot(h.history['val_' + metric], label='validation')
    plt.plot(h.history[metric], label='train')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.show()


def macro_roc(test_y, predictions, n_classes):
    """
    Computes the macro-averaged ROC curve for a classifier with more than 2 classes.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def rock_to_mineral_group(labels, index_file_path):
    """
    Args:
        labels: np.array containing string of rock types
        index_file_path: path to the excel file to be used for the conversion of rock types to the corresponding 
                         mineral group
    Returns: np.array of size == labels.size where rock types have been converted to mineral groups
    """
    index_file = pd.read_excel(index_file_path, 'alle')
    new_labels = []
    for i in range(labels.size):
        if np.any(index_file['name'] == labels[i]):
            new_labels.append(
                index_file['mineralclass'][index_file['name'] == labels[i]].values[0])
        elif labels[i] == 'chalkosin':
            new_labels.append("sulfides")
        elif labels[i] == 'dioptase':
            new_labels.append("quartz and silicates")
        else:
            raise RuntimeError(
                "Expected to be one of the three above cases and wasn't")

    new_labels = np.array(new_labels)
    return new_labels
