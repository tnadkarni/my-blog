---
layout: post
title:  "Creating a Random Forest from scratch"
date:   2020-11-01 21:20
category: Projects
---

Tree-based methods segment the predictor space based on splitting rules and can be applied to both Classification and Regression. In terms of accuracy, these models tend to have high variance - this means that the results change significantly when using different training sets and thus are not very reliable for prediction. To overcome this, we use ensemble classifiers which fit many individual classifiers and combine the result to yield a single prediction. One such ensemble classifier is the Random Forest.

To gain a solid understanding of how Decision Trees and Random Forest works, we will build a Random Forest Classifier from scratch i.e. without making use of existing Machine Learning libraries. 

We will be using the [UCI Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)) and performing Binary Classification (0/1) to determine if a tumour is benign or malignant.

We make use of the ID3 Algorithm to build the decision trees. Read more about it [here](https://cis.temple.edu/~giorgio/cis587/readings/id3-c45.html). From the linked reference, the basic working of the ID3 algorithms is as follows-

1. In the decision tree, each node corresponds to an attribute, each arc to a possible value of that attribute
2. A leaf at any point specifies the expected value of the outcome for the records described from the root to that leaf
3. Each node in the tree should be associated with the attribute which is most informative among the attributes not yet considered. The informativeness of the attribute in determining the outcome is measured by <i>Entropy</i>

Let's go through the ID3 algorithm in detail as we define our DecisionTree class in which we add multiple methods.

To start, we define a Node class and initialize the tree to have a root node. 
We also define a <strong>learn</strong> method that calls on a recursive function <strong>growTree</strong> to train the tree and stores values within the tree

{% highlight python%}
class DecisionTree(object):
  class Node:

    def __init__(self, pred = None, l = None, r = None, val = None, res = None):
      self.l = l
      self.r = r
      self.val = val
      self.pred = pred
      self.res = res

  def __init__(self):
    self.tree = self.Node()
  
  def learn(self, X, y):
    X = np.array(X)
    y = np.array(y)
    y.shape = (y.shape[0], 1)
    rows = np.append(X,y,1)
    self.tree = self.growTree(rows)
    
{% endhighlight %}

Prior to building the <strong>growTree</strong> function, let's go over ID3 basics-<br>
Assume a training set T with n predictors. We are trying to predict a binary outcome 0/1 (Success/Failure) 
We randomly choose m predictors and iterate over them. For each predictor, choose the split that gives highest information gain. Return a tree with that predictor as node.

Here's the function definition of <strong>growTree</strong>-

Start by taking care of some exceptions-
1. If T is empty return single node with value 0/Failure
2. If T consists of obervations with a single outcome, return a node with that outcome value

{% highlight python%}
def growTree(self, rows):

      if len(rows) == 0:
     		return self.Node(res = np.array([0]))
        

      if len(set(rows[:,-1])) == 1:
        return self.Node(res = np.array([rows[0,-1]]))
        #if all class labels same, return root node set to that label

{% endhighlight %}

One feature that sets Random Forest apart from <strong>Bagging</strong> - another ensemble model combining several decision trees with the goal of reducing model variance - is the selection of a subset of predictors in each tree. This number (m) is typically assumed to be the square root of total number of predictors. 

While iterating over the m predictors we have used the following functions-
1. partition_classes - partition the dataset at a split point for a predictor
2. entropy - computes entropy; this is a measure of the disorder in a dataset

{% highlight python%}
      n_predictors = len(rows[0])-1

      #randomly choose m predictors
      m_predictors =  np.random.choice(n_predictors, size=int(round(sqrt(n_predictors))))

      max_gain = 0.0y
      best_pred = None
      best_sets = None

      #iterate over each of the m predictors and calculate information gain from splitting at every value for that predictor

      for pred in m_predictors:
        vals = sorted(set(rows[:,pred]))[:-1]
        prev_classes = rows[:,-1]
        for value in vals:
          part1, part2 = partition_classes(rows, pred, value)
          new_gain = information_gain(prev_classes, [part1[:,-1], part2[:,-1]])
          if new_gain > max_gain:
            max_gain = new_gain
            best_pred = [pred, value] #best predictor and value to split at
            best_sets = [part1, part2] #resulting partitions from best split

      if max_gain > 0.05: #build tree as long as there is significant information gain in partitioning
        lb = self.growTree(best_sets[0])
        rb = self.growTree(best_sets[1])

        return self.Node(pred = best_pred[0], val = best_pred[1], l = lb, r = rb)

      else:
        return self.Node(res = rows[:,-1]) #stop building tree and assign result to leaf node
{% endhighlight %}

Through each iteration of <strong>growTree</strong>, we uncover the predictor that delivers the highest information gain, the split point for that predictor and the left (less than split) and right (greater than split) resulting datasets. We define the functions used in a separate <i>util.py</i> file.

Entropy is a measure of randomness of the information being processed and is calculated as
$$ E(S) = \sum_{i=1}^c -p_ilog_2p_i $$

We can think of Entropy as a measure of purity of the dataset. If a categorical predictor can take 2 different values e.g. A and B, entropy is low when a predictor contains only one class (either A or B) and high when it contains an equal distribution of both A and B. 

Information Gain is calculated as decrease in entropy after a data-set split on an attribute, represented by the formula-<br>
$$ IG(Y,X) = E(Y) - E(Y|X) $$

This gives us the reduction in uncertainty of Y (whether tumour is belign or malignant) given the knowledge of X(predictor value e.g. Clump Thickness=2).
Constructing a decision tree is based on finding the attribute and split value that returns the highest information gain.

Define these functions in a python file called util.py - 

{% highlight python%}
from scipy import stats
import numpy as np
from math import log

def entropy(class_y):
    class_y = np.array(class_y)
    p1 = float(sum(class_y))/len(class_y)
    p0 = 1-p1
    try:
        h = -p1*log(p1,2) - p0*log(p0,2)
    except ValueError:
        h = 0
    return h

def partition_classes(rows, pred, split_point):
    #Partition the dataset by the split point for specified predictor   
    part1 = rows[rows[:, pred] <= split_point]
    part2 = rows[rows[:, pred] > split_point]
    return [part1, part2]
    
def information_gain(previous_y, current_y):
    new_ent = 0.0
    size  = len(previous_y)
    for i in range(len(current_y)):
        new_ent = new_ent + entropy(current_y[i])*len(current_y[i])/size
    return entropy(previous_y)-new_ent
{% endhighlight %}

Now all that's left to do is build a <strong>RandomForest</strong> class with methods for 
* Initializing the RandomForest object as a list of DecisionTree class objects
* Bootstrapping - a type of random sampling with replacement for creating a different dataset for each tree
* Fitting multiple decision trees by calling on the <strong>learn</strong> defined earlier
* Classifying an OOB (out of bag) sample for error estimation using the mode of observations (if this were a regressionp problem, we would go with the mean value)

{% highlight python%}
class RandomForest(object):
    num_trees = 0
    decision_trees = []
    bootstraps_datasets = []
    bootstraps_labels = []

    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]

    def _bootstrapping(self, XX, n):
        XX = np.array(XX)
        ind = np.random.randint(XX.shape[0], size=n)
        return XX[ind, :-1], XX[ind, -1]

    def bootstrapping(self, XX):
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        for i in range(self.num_trees):
            self.decision_trees[i].learn(self.bootstraps_datasets[i], self.bootstraps_labels[i])

    def voting(self, X):
        y = np.array([], dtype = int)

        for record in X:
            #find the sets of proper trees that consider the record as an out-of-bag sample, and predict the label(class) 
            #for the record. The majority vote serves as the final label for this record.
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record.tolist() not in dataset[:,:-1]:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)
            counts = np.bincount(votes[0])
            if len(counts) == 0:
                y = np.append(y,0)
            else:
                y = np.append(y, np.argmax(counts))
        return y
{% endhighlight %}

This can be used by the <strong>main</strong> function to accomplish the following
1. Initialize a RandomForest object and specify the number of trees used
2. Load the data and create the bootstrapped datasets
3. Build the decision trees/fit the model
4. Validate the model using an OOB error estimate

{% highlight python%}
def main():
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels

    # Load data set
    with open("data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter=","):
            X.append(line[:-1])
            y.append(line[-1])
            xline = [ast.literal_eval(i) for i in line]
            XX.append(xline[:])

    # Set number of decision trees to build the forest
    forest_size = 10

    # Initialize a random forest
    randomForest = RandomForest(forest_size)

    # Create the bootstrapping datasets
    randomForest.bootstrapping(XX)

    # Build trees in the forest
    randomForest.fitting()

    # Provide an unbiased error estimation of the random forest based on OOB estimate
    y_truth = np.array(y, dtype=int)
    X = np.array(X, dtype=float)
    y_predicted = randomForest.voting(X)
    results = [prediction == truth for prediction, truth in zip(y_predicted, y_truth)]

    # Calculate accuracy
    accuracy = float(results.count(True)) / float(len(results))
    oob = 1-accuracy
    print "accuracy: %.4f" % accuracy
    print "OOB error estimate: %.4f" % oob


if __name__ == "__main__":
    main()
{% endhighlight %}

I mentioned that Random Forest only considers a random subset of predictors (m < p) for each tree. This may seem odd but in reality targets the problem of high variance adeptly. Bagging (m = p), for instance, would create multiple correlated trees where the strongest predictors are always used which while giving us high quality individual trees does not do much to reduce variance for the ensemble. Random Forest overcomes this issue and makes the average of the resulting trees more reliable.

Thank you for reading. This programming assignment was submitted as coursework for <i>[CSE6242](http://poloclub.gatech.edu/cse6242/2016fall/) Data and Visual Analytics (Fall 2016), Georgia Tech College of Computing</i>. 

