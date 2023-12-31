% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layer_functions.R
\name{layer}
\alias{layer}
\title{dense layer}
\usage{
layer(
  n_out,
  activ_func,
  d_activ_func = NULL,
  bias = FALSE,
  init.func = NULL,
  init.vals = NULL,
  ...
)
}
\arguments{
\item{n_out}{number of output neurons}

\item{activ_func}{activation function. should be a vectorized function of a single argument \code{x}}

\item{d_activ_func}{derivative of activation function. Ideally, \code{activ_func} is differentiable by \code{\link[Deriv:Deriv]{Deriv::Deriv()}} so you do not need to manually specify the derivative.}

\item{bias}{Default FALSE. If TRUE, adds bias (intercept term) to the layer.}

\item{init.func}{function to use when setting initial values of parameters. If left NULL, will use default from \code{\link[=neural_network]{neural_network()}}}

\item{init.vals}{parameters given to \code{init.func}. If left NULL, will use default from \code{\link[=neural_network]{neural_network()}}}
}
\description{
Adds a dense layer to the neural network.
}
