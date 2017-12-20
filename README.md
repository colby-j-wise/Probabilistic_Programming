# Final Project Repository Template

This is the final project repository for Colby Wise and Michael Alvarino's
[Machine Learning with Probabilistic Programming](http://www.proditus.com/syllabus.html) final Project

## Problem Formulation
Can we use probabilistic programming to create a generative model of trip
durations for taxi trips in New York City?

## Dataset
We will be using the [NYC Yellow Cab
Dataset](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml),
specifically the data for 2016. This included approximately 1.8 million trips
between January and June. We initially preprocessed the data ourselves to add
neighborhoods of the pickups and dropoffs, but later found a copy of the
dataset that already included the preprocessing. The preprocessing we performed
initially is included in our `preprocessing` directory.

## Box's Loop 1
Our first iteration through Box's loop was as simple as we could try, a Baysian
Gaussian Linear Model such that $y = f(X) + b$ where $f(X) = WX$. We placed a
gaussian prior on both $W$ and $b$. The model was surprisingly accurate, but we
thought it could be improved by finidng some interaction between our features
with a basis function.

## Box's Loop 2
Our second iteration through Box's loop added a basis function to our gaussian
linear model. We decided to try a simple polynomial basis, so that $basis(X_i,
degree) = [X_i, X_i ^ 2, X_i^3, ..., X_i^degree]$, to the polynomial degree
specified. We found that this model improved accuracy, but as shown by our PPC
did not model our original data well.

## Box's Loop 3
Understanding that there were infinitely many basis functions and that we could
not check and test them all, we decided to use a gaussian process. We tested
two different kernels, the gaussian kernel, and the rational quadratic. We found
that the gaussian process, though not a more accurate prdictor of trip
duration, was a much better model of our generative process.

## Next Steps
Use tensorflow to optimize the parameters of the gaussian process kernel
function
