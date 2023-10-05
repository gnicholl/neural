
#' train a neural network
#'
#' This function is the workhorse of the package, doing the actual training of the model.
#' You should not ever have to use `nn_estimate` directly.
#'
#' @param nnobject a list of class "nn" containing the inputs gathered from each layer.
#' @param learnrate learning rate for gradient descent
#' @param gradmethod gradient descent method (currently not used)
#' @param nbatch randomly divides training examples into `nbatch` batches while training
#' @param nepoch number of passes through all training examples
#' @export
nn_estimate = function(nnobject,learnrate,gradmethod,nbatch,nepoch,...) {

  # init params
  param_W = NULL
  for (l in 1:nnobject$L) {
    if (l==1) {
      prev_dim = nnobject$d_in
    } else {
      prev_dim = nnobject$layers_dims[l-1]
    }
    if (nnobject$biases[l]) {
      prev_dim = prev_dim + 1
    }
    # call given randomization function and given parameters
    initparams = as.list(nnobject$layers_initvals[[l]])
    initparams$n = nnobject$layers_dims[l]*prev_dim
    initvals = do.call(nnobject$layers_initfunc[[l]], initparams)

    # set up matrix of weights with randomized values
    param_W[[l]] = matrix(initvals,
                          nrow=nnobject$layers_dims[l],
                          ncol=prev_dim)
  }

  thegradient = NULL
  init_gradient = function() {
    blah = NULL
    for (l in 1:nnobject$L) {
      if (l==1) {
        prev_dim = nnobject$d_in
      } else {
        prev_dim = nnobject$layers_dims[l-1]
      }
      if (nnobject$biases[l]) {
        prev_dim = prev_dim + 1
      }
      blah[[l]] = matrix(0,nrow=nnobject$layers_dims[l],ncol=prev_dim)
    }
    thegradient <<- blah
  }

  # dropout: randomly generate mask vectors
  mask_vectors = NULL
  setup_dropout = function() {
    tmp = NULL
    for (l in 1:nnobject$L){
      tmp[[l]] = rbinom(nnobject$layers_dims[l],1,1-nnobject$layers_dropout[l])
    }
    mask_vectors <<- tmp
  }

  fwdprop = function(i) {
    # forward
    h_layers = NULL
    z_layers = NULL
    penalty_term = 0
    for (l in 1:nnobject$L) {
      if (l==1) {
        prev_h = as.matrix(nnobject$x[i,])
      } else {
        prev_h = h_layers[[l-1]]
      }
      if (nnobject$biases[l]) {
        prev_h = rbind(prev_h,1)
      }
      z_layers[[l]] = param_W[[l]] %*% prev_h
      h_layers[[l]] = nnobject$activ_funcs[[l]](z_layers[[l]])
      h_layers[[l]] = h_layers[[l]]*mask_vectors[[l]]
      if (nnobject$lambda > 0) penalty_term = penalty_term + nnobject$lambda*sum(nnobject$penaltyfunc(param_W[[l]]))
    }

    lossfn =
      nnobject$loss_func(
        y     = nnobject$y[i,],
        ypred = nnobject$linkfunc(h_layers[[nnobject$L]])
      ) / nnobject$n
    return(lossfn + penalty_term)

  }

  nn_gradient = function(i) {
    # forward
    h_layers = NULL
    z_layers = NULL
    for (l in 1:nnobject$L) {
      if (l==1) {
        prev_h = as.matrix(nnobject$x[i,])
      } else {
        prev_h = h_layers[[l-1]]
      }
      if (nnobject$biases[l]) {
        prev_h = rbind(prev_h,1)
      }
      z_layers[[l]] = param_W[[l]] %*% prev_h
      h_layers[[l]] = nnobject$activ_funcs[[l]](z_layers[[l]])
      h_layers[[l]] = h_layers[[l]]*mask_vectors[[l]]
    }
    yhat = nnobject$linkfunc(h_layers[[nnobject$L]])
    #allyhat[i,] <<- yhat

    # backward
    jacobian = nnobject$d_linkfunc(yhat)
    jacobian = jacobian * matrix(1,nrow=nnobject$K,ncol=nnobject$K)
    deriv_loss = nnobject$d_loss_func(y=nnobject$y[i,],ypred=yhat) / nnobject$n

    deriv_W = NULL
    for (l in nnobject$L:1) {
      if (l==1) {
        prev_h = as.matrix(nnobject$x[i,])

      } else {
        prev_h = h_layers[[l-1]]
      }
      if (nnobject$biases[l]) {
        prev_h = rbind(prev_h,1)
      }

      if (l==nnobject$L) {
        cache = 0
        for (k in 1:nnobject$K) {
          cache = cache + deriv_loss[k]*jacobian[k,]
        }
        cache = cache*nnobject$d_activ_funcs[[l]](z_layers[[l]])
      } else {
        next_W = param_W[[l+1]]
        if (nnobject$biases[l+1]) {
          next_W = next_W[,1:(ncol(next_W)-1),drop=FALSE]
        }
        cache = ( t(next_W) %*% cache ) * nnobject$d_activ_funcs[[l]](z_layers[[l]])
      }

      deriv_W[[l]] = cache %*% t(prev_h)
      if (nnobject$lambda>0) deriv_W[[l]] = deriv_W[[l]] + nnobject$lambda*nnobject$d_penaltyfunc(param_W[[l]])
    }

    return( deriv_W )

  }

  # compute gradients, collect function value
  if (nbatch==1) {
    scramble = 1:nnobject$n
    batches = list("1"=1:nnobject$n)
  } else {
    scramble = sample(nnobject$n,size=nnobject$n,replace=FALSE)
    batches = split(1:nnobject$n, cut(1:nnobject$n,nbatch,labels = FALSE))
  }

  print("***INITIAL***")
  setup_dropout()
  print(sum(sapply(X=1:nnobject$n,fwdprop)))

  iter = 1
  while(iter <= nepoch) {
    print(paste0("****epoch ",iter,"****"))
    for (b in 1:nbatch) {
      setup_dropout()
      init_gradient()
      for ( i in scramble[batches[[b]]] ) {
        tmp = nn_gradient(i)
        for (l in 1:nnobject$L) {
          thegradient[[l]] = thegradient[[l]] + tmp[[l]]
        }
      }

      new_W = NULL
      for (l in 1:nnobject$L) {
        new_W[[l]] = param_W[[l]] - learnrate * thegradient[[l]]
      }
      param_W = new_W

      print(paste0("****batch ",b,"****"))
      print(sum(sapply(X=1:nnobject$n,fwdprop)))
    }

    iter = iter+1
  }

  nnobject$param_w = param_W
  nnobject$funcvalue = sum(sapply(X=1:nnobject$n,fwdprop))
  return(nnobject)
}
