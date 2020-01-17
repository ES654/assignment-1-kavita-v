import numpy as np
import pandas as pd

def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    counts = Y.value_counts(normalize=True)
    prob_list = list(counts)
    entropy = 0.0
    for p_class in prob_list:
        entropy+=(-(p_class)*np.log(p_class)/np.log(2))

    return entropy

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    counts = Y.value_counts(normalize=True)
    prob_list = list(counts)
    gini_index = 1.0
    for p_class in prob_list:
        gini_index-=p_class*p_class
    
    return gini_index

def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    attr_counts = attr.value_counts(normalize=True)
    attr_prob_list = list(attr_counts)
    info_gain = entropy(Y)
    for i in range(len(attr_prob_list)):
        Y_attr = []
        for j in range(len(attr)):
            if attr_counts.keys()[i]  == attr[j]:
                Y_attr.append(Y[j])
        info_gain -= attr_prob_list[i]*entropy(Y_attr)

    return info_gain

