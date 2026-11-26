# micrograd-rs

My own small library for AI written from scratch in rust

I wanted to do some project including AI, and learned many thing about the subject, wich made me want to learn more and experiment, so i created this to play around

## Features

- **Tensor**: 2D matrices with matmul, transpose, element-wise operations,
  broadcasting of a bias row, reductions. No dependency on autograd.
- **Var**: reverse-mode automatic differentiation on tensors.
  Operations: 
    - matmul
    - add
    - sub
    - mul_elem
    - scale
    - add_row
    - tanh
    - sigmoid
    - relu
    - log
    - sum
    - mean
- **Loss**: 
    - mean squared error (`mse_loss`)
    - other are comming
- **Scalar engine**: the original micrograd-style `Value` engine, kept for learning and experimenting with different architechture.

## Examples:

All example are in example/ maybe i will put link of project i made using this lib later

## Tests

Imade a big bunch of test, so it would be easier to debug, suprinsigly, i quite enjoyed making the test for all methematical function, property check especially.

## GOALS:

The main goal of this library is to have all kind of people to use it, to be as free and useable as possible.

also since i will probably be the only one using it, its mean to be upgraded along my need and idea, but if anyone want to contribute, your welcome !

