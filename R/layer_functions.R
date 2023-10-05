
#' initial setup for neural network
#'
#' This function should be the first called to set up the neural network. See examples.
#'
#' @param x input data
#' @param y target data
#' @param init.func default function to use when setting initial values of parameters. First argument of the function should be `n`.
#' @param init.vals parameters given to `init.func` (safest bet is to give named list, e.g. `list(min=-1,max=1)`)
#'
#' @export
neural_network = function(x,y,init.func=runif, init.vals=c(-1,1), ...) {
  x = as.matrix(x)
  y = as.matrix(y)
  object = list(
    type="neural_network",
    x=x,
    y=y,
    n=nrow(y),
    K=ncol(y),
    L=0,
    d_in=ncol(x),
    layers_dims=NULL,
    layers_dropout=NULL,
    layers_initfunc=NULL,
    layers_initvals=NULL,
    default_initfunc=init.func,
    default_initvals=init.vals,
    activ_funcs=NULL,
    d_activ_funcs=NULL,
    biases=NULL,
    linkfunc=function(x) x,
    d_linkfunc=function(x) 1,
    loss_func=NULL,
    d_loss_func=NULL,
    penaltyfunc=NULL,
    d_penaltyfunc=NULL,
    lambda=0,
    num_embeds = 0
  )
  class(object) = "nn"
  return(object)
}

#' dense layer
#'
#' Adds a dense layer to the neural network.
#'
#' @param n_out number of output neurons
#' @param activ_func activation function. should be a vectorized function of a single argument `x`
#' @param d_activ_func derivative of activation function. Ideally, `activ_func` is differentiable by [Deriv::Deriv()] so you do not need to manually specify the derivative.
#' @param bias Default FALSE. If TRUE, adds bias (intercept term) to the layer.
#' @param init.func function to use when setting initial values of parameters. If left NULL, will use default from [neural_network()]
#' @param init.vals parameters given to `init.func`. If left NULL, will use default from [neural_network()]
#'
#' @export
layer = function(n_out,activ_func,d_activ_func=NULL,bias=FALSE,init.func=NULL,init.vals=NULL,...) {

  fn = env_activation[[as.character(substitute(activ_func))]]
  if (is.null(fn)) fn = activ_func
  if (is.null(d_activ_func)) d_activ_func=Deriv(fn,drule=neural::drule)

  object = list(
    type="layer",
    n_out=n_out,
    activ_func=fn,
    d_activ_func=d_activ_func,
    bias=bias,
    init_func=init.func,
    init_vals=init.vals
  )
  class(object) = "nn"
  return(object)
}

#' add dropout to previous dense layer
#'
#' @param rate dropout rate
#'
#' @export
dropout = function(rate,...) {

  if (rate<0 | rate>1) stop("dropout rate should be between 0 and 1")

  object = list(
    type="dropout",
    dropout_rate=rate
  )
  class(object) = "nn"
  return(object)
}

#' link function (to output layer)
#'
#' link function transforms the last hidden layer into
#' required output format without adding any additional parameters.
#' e.g. softmax (the only link function used at the moment)
#'
#' @param f link function to use (right now only use alt_softmax)
#' @param df derivative of link function
#'
#' @export
link = function(f,df=NULL,...) {
  fn = env_link[[as.character(substitute(f))]]
  if (is.null(fn)) fn = f
  if (is.null(df)) df=Deriv(fn,drule=neural::drule)

  object = list(
    type="link",
    linkfunc=fn,
    d_linkfunc=df
  )
  class(object) = "nn"
  return(object)
}

#' loss function
#'
#' @param f loss function. should be a function of `y` (target value) and `ypred` (predicted value)
#' @param df derivative of loss function. Ideally, `f` is differentiable by [Deriv::Deriv()] so you do not need to manually specify the derivative.
#'
#' @export
loss = function(f,df=NULL,...) {

  fn = env_loss[[as.character(substitute(f))]]
  if (is.null(fn)) fn = f
  if (is.null(df)) df=Deriv(fn,x="ypred",drule=neural::drule)

  object = list(
    type="loss",
    f=fn,
    df=df
  )
  class(object) = "nn"
  return(object)
}

#' penalty function
#'
#' adds regularization term to the loss function
#'
#' @param f loss function. should be a vectorized function of a single argument `x`
#' @param df derivative of loss function. Ideally, `f` is differentiable by [Deriv::Deriv()] so you do not need to manually specify the derivative.
#' @param lambda (non-negative) coefficient on the penalty term (bigger means more penalized)
#'
#' @export
penalty = function(f,df=NULL,lambda=0,...) {

  if (lambda<0) stop("penalty: require positive lambda")

  fn = env_penalty[[as.character(substitute(f))]]
  if (is.null(fn)) fn = f
  if (is.null(df)) df=Deriv(fn,drule=neural::drule)

  object = list(
    type="penalty",
    f=fn,
    df=df,
    lambda=lambda
  )
  class(object) = "nn"
  return(object)
}

#' train neural network
#'
#' This function specifies training parameters.
#'
#' @param learnrate learning rate for gradient descent
#' @param gradmethod gradient descent method (currently not used)
#' @param nbatch randomly divides training examples into `nbatch` batches while training
#' @param nepoch number of passes through all training examples
#'
#' @export
train = function(learnrate,gradmethod="gradient descent",nbatch=1,nepoch,...) {
  object = list(
    type="train",
    learnrate=learnrate,
    gradmethod=gradmethod,
    nbatch=nbatch,
    nepoch=nepoch
  )
  class(object) = "nn"
  return(object)
}

#' specify neural network layers
#'
#' This function overloads `+` so that neural network layers can be specified
#' analogous to how `ggplot` syntax works.
#'
#' @export
"+.nn" = function(o1,o2) {
  if (o1$type != "neural_network") stop("must start with neural_network call")

  if (o2$type == "layer") {
    o1$layers_dims = c(o1$layers_dims, o2$n_out)
    o1$activ_funcs = c(o1$activ_funcs, o2$activ_func)
    o1$d_activ_funcs = c(o1$d_activ_funcs, o2$d_activ_func)
    o1$biases = c(o1$biases,o2$bias)
    o1$L = o1$L + 1
    o1$layers_dropout[o1$L] = 0
    o1$layers_initfunc[[o1$L]] = if(is.null(o2$init_func)) o1$default_initfunc else o2$init_func
    o1$layers_initvals[[o1$L]] = if(is.null(o2$init_vals)) o1$default_initvals else o2$init_vals
    return(o1)

  } else if (o2$type == "loss") {
    o1$loss_func = o2$f
    o1$d_loss_func = o2$df
    return(o1)

  } else if (o2$type == "train") {
    return(
      nn_estimate(nnobject=o1,
                  learnrate=o2$learnrate,
                  gradmethod=o2$gradmethod,
                  nbatch=o2$nbatch,
                  nepoch=o2$nepoch)
    )

  } else if (o2$type == "link") {
    o1$linkfunc = o2$linkfunc
    o1$d_linkfunc = o2$d_linkfunc
    return(o1)

  } else if (o2$type == "penalty") {
    o1$penaltyfunc = o2$f
    o1$d_penaltyfunc = o2$df
    o1$lambda = o2$lambda
    return(o1)

  } else if (o2$type == "dropout") {
    o1$layers_dropout[o1$L] = o2$dropout_rate
    return(o1)

  } else if (o2$type == "embedding") {
    o1$num_embeds = o1$num_embeds + 1
    o1$embed_dim[o1$num_embeds] = o2$d_embed
    o1$embed_combine[o1$num_embeds] = o2$combine
    o1$embed_window[o1$num_embeds] = ifelse(o2$window %% 2 == 0, o2$window+1, o2$window)
    o1$embed_initfunc[[o1$num_embeds]] = if(is.null(o2$init_func)) o1$default_initfunc else o2$init_func
    o1$embed_initvals[[o1$num_embeds]] = if(is.null(o2$init_vals)) o1$default_initvals else o2$init_vals
    if (is.null(o2$embed_data)) {
      nlag = floor(o2$window/2)
      o1$embed_data[[o1$num_embeds]] = tidyr::separate_rows(data=data.frame(sen_num=1:length(o2$s),word_lag0=o2$s),word_lag0,sep=" ") %>%
        dplyr::group_by(sen_num) %>%
        mutate(word_num=dplyr::row_number())
      for (v in (-nlag):nlag) {
        if (v!=0) {
          o1$embed_data[[o1$num_embeds]] = o1$embed_data[[o1$num_embeds]] %>%
            dplyr::group_by(sen_num) %>%
            mutate("word_lag{v}" := help_lag(word_lag0,n=v,default="<NULL>"))
        }
      }
      return(o1)

    } else {
      o1$embed_data[[o1$num_embeds]] = o2$embed_data
    }

  } else {
    stop("mistake was made")
  }

}


