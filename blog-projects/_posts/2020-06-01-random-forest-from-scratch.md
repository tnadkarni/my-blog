---
layout: post
title:  "Creating a Random Forest from scratch"
date:   2020-06-01 21:20
category: Projects
---

Thanks to ready-to-use packages like [Scikit-Learn](https://scikit-learn.org/stable/) we can implement numerous Machine Learning algorithms easily. This black box implementation while quick and highly functional, does not allow us to see what actually happens under the hood and for those of us starting out with Machine Learning, using these libraries without knowing the concept is unadvisable.

To make sure we have the correct understanding, we will build a Random Forest Classifier from scratch i.e. without making use of existing Machine Learning libraries. 

We will be using the [UCI Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)) and performing Binary Classification (0/1) to determine if a tumour is benign or malignant.

A Random Forest consists of multiple Decision Trees. It is what is known as an Ensemble classifier which fits many individual classifiers through a weighted combination and is typically more accurate than using a single classifier. A Decision Tree is a supervised learning algorithm which can be applied to Classification or Regression problems.To put it simply, it consists of a graphical representation of possible solutions starting with a root node followed by splits to leaf nodes depending on different conditions for attributes at each node. 

We will make use of the ID3 Algorithm to build the decision trees. Read more about it [here](https://cis.temple.edu/~giorgio/cis587/readings/id3-c45.html). From the linked reference, the basic working of the ID3 algorithms is as follows-

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

Prior to building the <strong>growTree</strong> function, let's go over ID3 basics.
Assume a training set T with n predictors. We are trying to predict a binary outcome 0/1 (Success/Failure) 
We randomly choose m predictors and iterate over them. For each predictor, choose the split that gives highest information gain. Return a tree with that predictor as node.
Information Gain is calculated as decrease in entropy after a data-set split on an attribute. Constructing a decision tree is based on finding the attribute that returns the highest information gain.
Entropy is a measure of randomness of the information being processed and is calculated as
$$ 
E(S) = \sum_{i=1}^c -p_ilog_2p_i
$$


Start by taking care of some exceptions-
1. If T is empty return single node with value 0/Failure
2. If T consists of obervations with a singel outcome, return a node with that outcome value

{% highlight python%}
def growTree(self, rows):
      # this recursive function builds the tree

      if len(rows) == 0:
     		return self.Node(res = np.array([0]))
        #if dataset is empty return tree with just root node set to 0 (failure)

      if len(set(rows[:,-1])) == 1:
        return self.Node(res = np.array([rows[0,-1]]))
        #if all class labels same, return root node set to that label
{% endhighlight %}


Lets



We randomly choose m predictors and iterate over them. For each predictor, choose the split that gives highest information gain. 



{% highlight python%}
      n_predictors = len(rows[0])-1

      #randomly choose m = sqrt(n) predictors
      m_predictors =  np.random.choice(n_predictors, size=int(round(sqrt(n_predictors))))

      max_gain = 0.0
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







