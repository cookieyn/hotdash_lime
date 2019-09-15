import json
import copy
from sklearn.tree._tree import TREE_LEAF
from math import sqrt
import numpy as np
import math
import time


def rules(clf, features, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1: # leaf nodes
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        node['name'] = 'leaf'
    else:
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} <= {}'.format(feature, threshold)
        left_index = clf.tree_.children_right[node_index]
        right_index = clf.tree_.children_left[node_index]
        node['children'] = [rules(clf, features, right_index),
                            rules(clf, features, left_index)]
    return node


def Rt_compute(Rt_compute):
    # 1. Accuracy
    # Rt=(sum(model['value'])-max(model['value']))
    # return Rt
    # 2. Gini loss
    sq_count = 0.0
    samples = np.sum(Rt_compute['value'])
    for k in Rt_compute['value']:
        sq_count += k * k
    gini = 1.0 - sq_count / (samples * samples)
    return samples * gini


def RTt_compute(model, leaves_error_count):
    if "children" not in model:
        return Rt_compute(model)
    children = model['children']
    for child in children:
        leaves_error_count += RTt_compute(child, 0)
    return leaves_error_count


def gt_compute(json_tree):
    return (Rt_compute(json_tree)-RTt_compute(json_tree, 0)*1.0) / (str(json_tree).count('leaf') - 1)


def gt_with_tree(json_tree, gt_list, path, path_list):
    if 'children' not in json_tree:
        return
    else:
        gt_list.append(gt_compute(json_tree))
        path_list.append(path)
        children = json_tree['children']
        for i in range(len(children)):
            gt_with_tree(children[i], gt_list, path+[i], path_list)


def prune_sklearn_model(sklearn_model, index, json_model):
    if "children" not in json_model:
        # Modify values of leaf nodes
        for i in range(len(json_model['value'])):
            sklearn_model.value[index][0][i] = json_model['value'][i]
        sklearn_model.children_left[index] = TREE_LEAF
        sklearn_model.children_right[index] = TREE_LEAF
    else:
        prune_sklearn_model(sklearn_model, sklearn_model.children_left[index], json_model['children'][0])
        prune_sklearn_model(sklearn_model, sklearn_model.children_right[index], json_model['children'][1])


def model_gtmin_Tt(json_tree):    # T0->T1
    gt_list = []
    path = [1]
    path_list = []
    gt_with_tree(json_tree, gt_list, path, path_list)
    alpha = min(gt_list)
    prune_gt_index = gt_list.index(alpha)
    # Delete child by child-path
    temp_tree = json_tree
    for i in path_list[prune_gt_index][1:]:
        temp_tree = temp_tree['children'][i]
    del temp_tree['children']
    temp_tree['name'] = 'leaf'

    return json_tree, alpha, path_list[prune_gt_index]


def candidate(tree, json_model, alpha_list, tree_list, max_leaf_nodes):
    alpha = 0
    json_tree = copy.deepcopy(json_model)

    while True:
        leaf_num = str(json_tree).count('leaf')
        if max_leaf_nodes is None or leaf_num < max_leaf_nodes:
            alpha_list.append(alpha)
            temp_tree = copy.deepcopy(tree)
            prune_sklearn_model(temp_tree.tree_, 0, json_tree)
            tree_list.append(temp_tree)

        json_tree, alpha, _ = model_gtmin_Tt(json_tree)

        if "children" not in json_tree:
            alpha_list.append(alpha)
            temp_tree = copy.deepcopy(tree)
            prune_sklearn_model(temp_tree.tree_, 0, json_tree)
            tree_list.append(temp_tree)
            break

    return alpha_list, tree_list


def validate(TreeSets, alpha_list, x_test, y_test):
    precision_list = []
    for item in TreeSets:
        precision_list.append(np.mean(item.predict(x_test) == y_test))

    max_precision = max(precision_list)
    index_error_rate = precision_list.index(max_precision)

    # Add for test
    pruned_precision = precision_list[
        index_error_rate]  # here's right,because the precision list is corresponding to the error_rate_list.
    best_alpha = alpha_list[index_error_rate]
    Best_tree = TreeSets[index_error_rate]
    print('During pruning, best tree is', index_error_rate)
    print('Node Count', Best_tree.tree_.node_count)
    print('best_alpha', best_alpha, 'pruned_precision', pruned_precision)
    return Best_tree


def prune(tree, x_test, y_test, max_leaf_nodes):
    features = ['X' + str(i) for i in range(tree.n_features_)]
    json_model = rules(tree, features)

    alpha_list, tree_list = candidate(tree, json_model, [], [], max_leaf_nodes)
    best_tree = validate(tree_list, alpha_list, x_test, y_test)
    return best_tree
