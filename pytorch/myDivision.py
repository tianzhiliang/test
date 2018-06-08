from __future__ import print_function
import torch
import copy

def myMatrixDivVector(matrix, vector):
    """
       matrix(N,M) / vector(N) = matrix(N,M)
       for each i,j: 
           matrix_result[i][j] = matrix_source[i][j] / vector[i]
    """
    matrix1 = torch.transpose(matrix, 0, 1)
    print("matrix transpose:", matrix1)
    for i, nm in enumerate(matrix1):
        matrix1[i] = nm / vector
    print("matrix after division:", matrix1)
    matrix = torch.transpose(matrix1, 0, 1)
    print("matrix (final result):", matrix)
    return matrix

def myMatrixDivVector2(matrix, vector):
    """
       matrix(N,M) / vector(N) = matrix(N,M)
       for each i,j: 
           matrix_result[i][j] = matrix_source[i][j] / vector[i]
    """
    duplicate_size = matrix.size()[1]
    print("duplicate_size:", duplicate_size)
    vector_duplicate = vector.repeat(duplicate_size, 1).permute(1, 0)
    print("vector_duplicate:", vector_duplicate)
    matrix = matrix / vector_duplicate
    print("matrix (final result):", matrix)
    return matrix

a = torch.rand(4, 3)
b = torch.rand(4)
a2 = copy.deepcopy(a)
b2 = copy.deepcopy(b)
print("a:", a)
print("b:", b)
myMatrixDivVector(a, b)
myMatrixDivVector2(a2, b2)
