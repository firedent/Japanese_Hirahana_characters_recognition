#!/usr/bin/env python3
"""
part1.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""
import torch


# Simple addition operation

def simple_addition(x, y):
    return torch.add(x, y)


# Resize tensors
# Use view() to implement the following functions ( flatten() and reshape() are not allowed )

def simple_reshape(x, shape):
    return x.view(shape)


def simple_flat(x):
    return x.view(x.size().numel())


# Transpose and Permutation

def simple_transpose(x):
    return x.transpose(0, 1)


def simple_permute(x, order):
    return x.permute(order)


# Matrix multiplication (with broadcasting).

def simple_dot_product(x, y):
    return torch.dot(x, y)


def simple_matrix_mul(x, y):
    return torch.matmul(x, y)


def broadcastable_matrix_mul(x, y):
    return torch.matmul(x, y)


# Concatenate and stack.
def simple_concatenate(tensors):
    return torch.cat(tensors)


def simple_stack(tensors, dim):
    return torch.stack(tensors, dim)
