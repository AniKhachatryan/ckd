"""
"""

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


def evaluate(y_true, y_predicted):

    # assert that the lengths of y_true and y_predicted are equal
    assert len(y_true) == len(y_predicted), 'y_true and y_predicted lengths should match.'

    # compute performance metrics
    accuracy = accuracy_score(y_true, y_predicted)
    recall_sensitivity = recall_score(y_true, y_predicted, pos_label='ckd')
    specificity = recall_score(y_true, y_predicted, pos_label='notckd')
    precision = precision_score(y_true, y_predicted, pos_label='ckd')
    confusion_mat = confusion_matrix(y_true, y_predicted)

    # create a dict
    performance_metrics = {
        'y_true': y_true,
        'y_predicted': y_predicted,
        'accuracy': accuracy,
        'recall_sensitivity': recall_sensitivity,
        'specificity': specificity,
        'precision': precision,
        'confusion_mat': confusion_mat
    }

    # print some metrics
    print(f'Classification performance metrics:\n'
          f'Accuracy: {accuracy}\n'
          f'Recall (sensitivity): {recall_sensitivity}\n'
          f'Specificity: {specificity}\n'
          f'Precision: {precision}\n')

    return performance_metrics
