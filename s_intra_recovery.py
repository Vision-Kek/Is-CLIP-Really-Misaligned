# This script walks through the procedure from Appendix A step by step.
# It demonstrates how recovery of image-image similarities is possible given only text-image similarities.

import torch
import torch.nn.functional as F

########################
#   Main Procedure     #
########################

N, d = 42, 3  # You can set the number of embeddings (N) and their dimension here (d)
assert N > d*(d+1)/2, f'{N}<{d*(d+1)/2}, make sure N > d(d+1)/2' 

# Setup
X_T = F.normalize(torch.rand(N,d),dim=1)  # text embeddings
X_I = F.normalize(torch.rand(N,d),dim=1)  # image embeddings
S_inter = X_T @ X_I.T  # text-image similarities

# Decompose
U, Sigma, Vt = torch.linalg.svd(S_inter)
U, V = U[:,:d], Vt.T[:,:d]  # reduced SVD (rank d)

pairwise_products = V.unsqueeze(2) * V.unsqueeze(1)

# Solve
A = pairwise_products.view(N,d*d)
b = torch.ones(N)

x = torch.linalg.lstsq(A, b, driver='gelsd').solution

# Recover
Q = x.view(d,d)
S_intra_recovered = V @ Q @ V.T

# Check
S_intra_true = X_I @ X_I.T
print('RESULT: recovery error:', (S_intra_recovered - S_intra_true).abs().max())


########################
#      Analysis        #
########################
print('-----------')
print('ANALYSIS')
print('1. Validate "The columns of U span the same space as the columns of X_I":')

def equal_column_space(A, B):
    # Theorem: The column spaces of A and B are equal if and only if the ranks of the following three matrices are the same:
    # rank(A)=rank(B)=rank([A ∣ B])
    rank_A = torch.linalg.matrix_rank(A)
    rank_B = torch.linalg.matrix_rank(B)
    rank_A_aug_B = torch.linalg.matrix_rank(torch.cat([A,B],dim=1))
    return rank_A == rank_B == rank_A_aug_B

print(f'{equal_column_space(X_T, S_inter)=}, True expected') # column space of S_inter is the columns space of X_T
print(f'{equal_column_space(U, S_inter)=}, True expected') #  X_T -svd-> U
print(f'{equal_column_space(U, X_T)=}, True expected')

print(f'{equal_column_space(X_I, S_inter.T)=}, True expected') # the row space of S_inter is the column space of
print(f'{equal_column_space(V, S_inter.T)=}, True expected') #  X_I -svd-> V
print(f'{equal_column_space(V, X_I)=}, True expected')

print(f'{equal_column_space(V, S_inter)=}, False expected') # sanity check

print('-----------')
print('2. Validate "$b = \mathbf{1}_N$ is in column space of $A$":')

def in_column_space(A, b):
    # Theorem: b is in the column space of A if and only if the ranks of the following two matrices are the same:
    rank_A = torch.linalg.matrix_rank(A)
    rank_A_aug_b = torch.linalg.matrix_rank(torch.cat([A,b[:,None]],dim=1))
    return rank_A == rank_A_aug_b

# setup A
# it's the pairwise product of all column combinations in V
A = (V[:,None,:] * V[:,:,None]).reshape(N, d*d)
print(f'{torch.linalg.matrix_rank(A)=}, expected {d*(d+1)/2}')

# " b = 1_N is in column space of A "
b = torch.ones(N)
print(f'{in_column_space(A, torch.ones(N))=}, True expected')
# Why?
# Because
# - A is built from V.
# - From the previous validation, we know the column space of V and X_I are the same.
A_with_X_I_instead_of_V = (X_I[:,None,:] * X_I[:,:,None]).reshape(N, d*d)
print(f'{in_column_space(A_with_X_I_instead_of_V, b)=}, True expected')
# now why is b in the columns of X_I ?
# -> because X_I is normalized, such that a combination (specifically, the sum) of all columns in X_I**2 is 1:
print(f'{(X_I**2).sum(axis=-1)=}')
# hence:
print(f'{in_column_space(X_I**2, b)=}, True expected')
# hence also the diagonals in A corresponding to the self-squares sum to 1
print((A_with_X_I_instead_of_V * torch.eye(d).flatten()).sum(axis=1))
# such that b is in the column space of A_with_X_I_instead_of_V
# and since A_with_X_I_instead_of_V and A share the column space, also in the column space of A.
print(f'{equal_column_space(A_with_X_I_instead_of_V, A)=}, True expected') # sanity check
