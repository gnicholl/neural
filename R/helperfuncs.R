
#' @import Deriv
mysum = function(x) sum(x)
drule$mysum = alist(x=1)
mysum2 = function(x) sum(x)
drule$mysum2 = alist(x=x)

# activation functions
  env_activation = new.env()

  # relu
  env_activation$relu = function(x) {
    ifelse(x>0,x,0)
  }
  env_activation$ReLU = env_activation$relu

  # sigmoid
  env_activation$sigmoid = function(x) {
    1/(1+exp(-x))
  }

  # identity
  env_activation$identity = function(x) {
    x
  }

  # tanh
  env_activation$tanh = tanh

  # hard tanh
  env_activation$hardtanh = function(x) {
    ifelse(x<-1,-1,ifelse(x>1,1,x))
  }

# loss functions (must be function of y and ypred) (NLP book Ch2.7.1)
  env_loss = new.env()

  # L2
  env_loss$L2 = function(y,ypred) {
    0.5*(ypred-y)^2
  }

  # logistic
  env_loss$binom = function(y,ypred) {
    -y*log(ypred) - (1-y)*log(1-ypred)
  }
  env_loss$logistic = env_loss$binom

  # multinomial
  env_loss$multinom = function(y,ypred) {
    -mysum(y*log(ypred+1e-10))
  }
  env_loss$alt_multinom = function(y,ypred) {
    -mysum(y*log(ypred+1e-20)) - (1-sum(y))*log(1-mysum(ypred+1e-20))
  }

# penalty functions
  env_penalty = new.env()

  env_penalty$L2 = function(x) {
    0.5*x^2
  }

# link functions
  env_link = new.env()

  help_softmax = function(x) {
    exp(x-max(x)) / sum(exp(x-max(x)))
  }
  env_link$softmax = function(x) {
    help_softmax(x)
  }
  drule$help_softmax = alist(x=diag(x) - x %*% t(x))

  alt_help_softmax = function(x) {
    exp(x-max(0,x)) / (exp(0-max(0,x)) + sum(exp(x-max(0,x))))
  }
  env_link$alt_softmax = function(x) {
    alt_help_softmax(x)
  }
  drule$alt_help_softmax = alist(x=diag(x) - x %*% t(x))




# miscellaneous
  help_lag = function(x, n=1L, default=NULL, order_by=NULL) {
    if (n>0) return(dplyr::lead(x=x,n=n,default=default,order_by=order_by))
    if (n<0) return(dplyr::lag(x=x,n=abs(n),default=default,order_by=order_by))
    stop("whoops")
  }






