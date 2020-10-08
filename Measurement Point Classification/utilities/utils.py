import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.utils import to_categorical


def plot_confusion_matrix(cm, classes, normalize=False, fp_precision=6,
                          title='Confusion matrix', cmap=plt.cm.Blues,
                          save_path=None):
    """
    Args:
        cm: confusion matrix to be plotted. Most likely it will be the output of sklearn's confusion_matrix function.
        classes: list of classes that correspond to the columns and rows of the confusion_matrix
        normalize: bool, whether to normalize or not the input confusion matrix
    """
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)
                      [:, np.newaxis], decimals=fp_precision)
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

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
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


def to_one_hot_encoding(labels):
    """
    The order for the one-hot encoded entries is determined by numpy's 
    default sorting.
    """
    classes = np.unique(labels)
    y = np.zeros(len(labels))
    for i in range(classes.size):
        y[labels == classes[i]] = i

    y = to_categorical(y)

    return y


def pred_to_conf_matrix(pred_path, cv_len, n_classes):
    pred_array = np.load(pred_path, allow_pickle=True)
    assert pred_array.shape[0] == cv_len

    labels = np.arange(n_classes)

    confusion_matrices = []
    for element in pred_array:
        if 'major_vote' in pred_path:
            predictions = np.array(element[0])
        else:
            predictions = np.argmax(element[0], axis=1)

        true_labels = np.argmax(
            np.array([element[1] for _ in range(predictions.size)]), axis=1)

        cm = confusion_matrix(true_labels, predictions, labels=labels)
        confusion_matrices.append(cm)

    confusion_matrices = np.array(confusion_matrices)
    assert confusion_matrices.shape == (cv_len, n_classes, n_classes)

    return np.mean(confusion_matrices, axis=0)


def pred_to_roc_curves(preds, pred_labels):
    """
    Args:
            preds: np.array of predictions, shape (n_sample, n_classes)
            pred_labels: np.array of one-hot encoded labels, shape (n_samples, n_classes)
    Returns:
            roc_curves: list of length = n_classes containing all relevant info to plot ROC curves for each class
    """
    classes = np.arange(preds.shape[1])

    roc_curves = []

    for label in classes:
        predictions = []
        true_labels = []
        for i in range(preds.shape[0]):
            predictions.append(preds[i][label])
            true_labels.append(pred_labels[i][label])

        predictions = np.array(predictions).flatten()
        true_labels = np.array(true_labels).flatten()

        fpr, tpr, thresholds = roc_curve(true_labels, predictions)

        roc_curves.append([fpr, tpr, thresholds])

    return roc_curves


