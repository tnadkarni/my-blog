---
layout: post
title:  "Creating a Random Forest from scratch"
date:   2020-06-01 21:20
category: Projects
---

Thanks to ready-to-use packages like [Scikit-Learn](https://scikit-learn.org/stable/) we can implement numerous Machine Learning algorithms without going into details of how it actually works. This black box implementation while quick and easy, does not allow us to see what actually happens under the hood and for those of us starting out with Machine Learning, using these libraries without knowing the concept is unadvisable.

To make sure we have the correct understanding, we're going to build a Random Forest Classifier from scratch i.e. without making use of existing Machine Learning libraries. 

We will be using the [UCI Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)) and performing Binary Classification (0/1) to determine if a tumour is benign or malignant.

To start with, let's define the structure. To implement a Random Forest we would need-
1. decision_tree.py:
A Random Forest consists of multiple Decision Trees. It is what is known as an Ensemble classifier which fits many individual classifiers through a weighted combination and is typically more accurate than using a single classifier. Let's build a class for this classifier (decision tree)

Let's recap - a Decision Tree is a supervised learning algorithm which can be applied to Classification or Regression problems.To put it simply, it consists of a graphical representation of possible solutions starting with a root node followed by splits to leaf nodes depending on different conditions for attributes at each node. 

![image](/blog-projects/assets/images/ex_dtree.jpg) Img Source:hackerearth.com

We will follow the [ID3](https://cis.temple.edu/~giorgio/cis587/readings/id3-c45.html) Algorithm. In order 

The ID3 algorithm follows 


{% highlight python%}
class DecisionTree(object):
#create node class
    class Node:
      def __init__(self, pred = None, l = None, r = None, val = None, res = None):
        self.l = l
        self.r = r
        self.val = val
        self.pred = pred
        self.res = res
    
{% endhighlight %}

2. util.py
3. random_forest.py










