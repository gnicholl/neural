# neural: Simple Neural Network in R

Simple neural network implemented in R, with ggplot2-inspired syntax, e.g.:

```
model = 
  neural_network(x=x,y=y) +
   layer(8,tanh, bias=TRUE, init.func=runif, init.vals=c(0,1)) +
   dropout(0.25) +
   layer(2,identity, bias=TRUE, init.func=runif, init.vals=c(0,1)) +
   link(softmax) +
   loss(multinom) +
   penalty(L2,lambda=0.0005)

estmodel = model +
  train(learnrate=0.05,nepoch=20,nbatch=2)
```
