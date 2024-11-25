# Imports
from typing import Dict
import tensorflow as tf

# Further imports are NOT allowed, please use the APIs in `tf`, `tf.keras` and `tf.keras.backend`!


def recall(matrix: tf.Tensor, idx: int) -> tf.Tensor:
    """Calclate the recall metric for a given confusion matrix and category.

    Arguments
    ---------
    matrix : tensorflow.Tensor
        The confusion matrix for the trained model. (rows: real labels, cols: predicted labels)
    idx : int
        The category index (0: silence, 1: unknown, 2: ...)

    Returns
    -------
    recall : tensorflow.Tensor
        The calculated recall value (between 0 and 1)
    """

    recall = None

    ### ENTER STUDENT CODE BELOW ###
    '''
    confusion matrix : 
                        predicted positive      predicted negative
    actual positive     true_positive           false_negative
    actual negative     false_positive          false_negative
    
    true_positive(TP)   : Model correctly identifies a positive instance
    false_positive(FP)  : Model incorrectly identifies a negative instance as positive
    false_negative(FN)  : Model incorrectly identifies a positive instance as negative
    true_negative(TN)   : Model correctly identifies a negative instance
    
    recall : how well a model can identify all the positive instances of a given class
    recall = TP/(TP+FN)
    
    precision :  how many of the instances predicted as positive are actually positive
    precision = TP/(TP+FP)
    
    f1_score : the harmonic mean of precision and recall
    f1_score = 2*(recall*precision)/(recall+precision)
    '''
    
    # ensure all datatypes are set to float32
    true_positive = tf.cast(matrix[idx, idx], tf.float32)
    false_negative = tf.cast(tf.reduce_sum(matrix[idx, :]), tf.float32) - true_positive
     # Adding `tf.keras.backend.epsilon()` to prevent division by zero
    epsilon = tf.keras.backend.epsilon()
    recall = true_positive / (true_positive + false_negative + epsilon)

    ### ENTER STUDENT CODE ABOVE ###

    return recall


def precision(matrix: tf.Tensor, idx: int) -> tf.Tensor:
    """Calclate the precision metric for a given confusion matrix and category.

    Arguments
    ---------
    matrix : tensorflow.Tensor
        The confusion matrix for the trained model. (rows: real labels, cols: predicted labels)
    idx : int
        The category index (0: silence, 1: unknown, 2: ...)

    Returns
    -------
    recall : tensorflow.Tensor
        The calculated precision value (between 0 and 1)
    """

    precision = None

    ### ENTER STUDENT CODE BELOW ###
    true_positive = tf.cast(matrix[idx, idx], tf.float32)
    false_positive = tf.cast(tf.reduce_sum(matrix[:, idx]), tf.float32) - true_positive
    epsilon = tf.keras.backend.epsilon()
    precision = true_positive / (true_positive + false_positive + epsilon)

    ### ENTER STUDENT CODE ABOVE ###

    return precision


def f1_score(matrix: tf.Tensor, idx: int) -> tf.Tensor:
    """Calclate the f1_score metric for a given confusion matrix and category.

    Arguments
    ---------
    matrix : tensorflow.Tensor
        The confusion matrix for the trained model. (rows: real labels, cols: predicted labels)
    idx : int
        The category index (0: silence, 1: unknown, 2: ...)

    Returns
    -------
    recall : tensorflow.Tensor
        The calculated f1_score value (between 0 and 1)
    """

    f1_score = None

    ### ENTER STUDENT CODE BELOW ###
    rec = recall(matrix, idx)
    prec = precision(matrix, idx)
    epsilon = tf.keras.backend.epsilon()
    f1_score = 2 * (rec * prec) / (rec + prec + epsilon)

    ### ENTER STUDENT CODE ABOVE ###

    return f1_score


def get_student_metrics(matrix: tf.Tensor, idx) -> Dict[str, tf.Tensor]:
    ret = {
        "recall": recall(matrix, idx),
        "precision": precision(matrix, idx),
        "f1_score": f1_score(matrix, idx),
    }
    return {key: value.numpy() for key, value in ret.items() if value is not None}
