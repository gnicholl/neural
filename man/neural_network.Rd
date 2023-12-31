% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layer_functions.R
\name{neural_network}
\alias{neural_network}
\title{initial setup for neural network}
\usage{
neural_network(x, y, init.func = runif, init.vals = c(-1, 1), ...)
}
\arguments{
\item{x}{input data}

\item{y}{target data}

\item{init.func}{default function to use when setting initial values of parameters. First argument of the function should be \code{n}.}

\item{init.vals}{parameters given to \code{init.func} (safest bet is to give named list, e.g. \code{list(min=-1,max=1)})}
}
\description{
This function should be the first called to set up the neural network. See examples.
}
