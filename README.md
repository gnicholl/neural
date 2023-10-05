# neural: Simple Neural Network in R

Simple neural network implemented in R, with ggplot2-inspired syntax.
For example, here is a multi-class classification problem:

```
# data
y = cbind(
  as.numeric(iris$Species=="setosa"),
  as.numeric(iris$Species=="versicolor"),
  as.numeric(iris$Species=="virginica"))
y = y[,2:3]
x = as.matrix(iris[,c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")])
x[,1] = (x[,1] - mean(x[,1])) / sd(x[,1])
x[,2] = (x[,2] - mean(x[,2])) / sd(x[,2])
x[,3] = (x[,3] - mean(x[,3])) / sd(x[,3])
x[,4] = (x[,4] - mean(x[,4])) / sd(x[,4])

# set seed
set.seed(12325)

# model specification
model = 
  neural_network(x=x,y=y) +
  layer(8,tanh, bias=TRUE, init.func=runif, init.vals=c(0,1)) +
  dropout(0.1) +
  layer(2,identity, bias=TRUE, init.func=runif, init.vals=c(0,1)) +
  link(alt_softmax) +
  loss(alt_multinom) +
  penalty(L2,lambda=0.0001)

# train model
estmodel = model +
  train(learnrate=0.05,nepoch=20,nbatch=1)
```
