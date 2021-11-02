# Outline

Problem: Want to identify what embeddings are useful for training and various vision tasks. We hypothesize that embeddings that more closely match neural representations are more powerful and generalizable.

Motivation: More deeply understand why our brain's visual neural representations are so powerful and capable of solving a wide range of tasks with limited data and training

Related work:  https://www.pnas.org/content/118/3/e2014196118.short
PCA
Basic MLPs

Methods: 

1. Download data
2. Choose a few embedding methods, train them via unsupervised learning
3. Use these embeddings as inputs to a separate model (which could be a neural net, or KNN or wtv) and look at the performance
