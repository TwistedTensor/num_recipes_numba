import numpy as np
from numba import *
@jit
def _SWAPROW(a,i,j):
    temp = a[i,:].copy()
    a[i,:] = a[j,:]
    a[j,:] = temp
    
@jit
def _SWAPCOL(a,i,j):
    temp = a[:,i].copy()
    a[:,i] = a[:,j]
    a[:,j] = temp

@jit(void(double[:,:],int32,double[:,:],int32))
def gaussj(a,n,b,m):
    indxc = np.arange(n) # ith entry will be the ith pivot column
    indxr = np.arange(n) # ith entry will be the ith pivot row
    ipiv = np.zeros(n)   # ith entry will be one if we have pivoted the ith column, else 0
    
    icol = irow = 0
    big = dum = pivint = temp = 0.
    for i in range(n):
        big = 0.
        # search for pivot element, want the largest element from a column
	# that has not already been pivoted
        for j in range(n):
            if ipiv[j] != 1:
                for k in range(n):
                    if (ipiv[k] == 0) & (abs(a[j,k]) >= big):
                        big = abs(a[j,k])
                        irow,icol = (j,k)
        ipiv[icol]+=1 # Record the fact that we have pivoted this column
	# Swap the rows:
        if irow != icol: 
            _SWAPROW(a,irow,icol)
            _SWAPROW(b,irow,icol)

        # Make note of the pivot row/column 
        indxr[i] = irow
        indxc[i] = icol

	# After the row swap, th pivot will be on the diagonal at [icol,icol]
	# If it is zero, then the matrix is singular and we will raise an exception
        if a[icol,icol]==0.:
            raise Exception('gaussj: Singular Matrix')

        pivinv = 1./a[icol,icol]

	# This is where we begin replacing the pivot column with what will eventually
	# become the inverse. Right now it is the identity matrix, so the diagonal element
	# [icol,icol] is 1.
        a[icol,icol] = 1.

	# Divide by the pivot element (multiply by the inverse)
        a[icol,:] *= pivinv
        b[icol,:] *= pivinv
        
        for ll in range(n):
            if ll != icol:
                dum = a[ll,icol] # This is the value we will use in the elimination step

		# Here are the rest of the replacements of the pivot column, the matrix
		# which will become the inverse is almost the identity matrix except for
		# the [icol,icol] element which was set to pivinv when we multiplied
		# the whole icol row by pivinv. Everything else in the icol column is zero:
                a[ll,icol] = 0.

		# Now we eliminate
                a[ll,:] -= a[icol,:]*dum
                b[ll,:] -= b[icol,:]*dum
                
    # We swaped all of the rows in the actual matrices, but the column swpas were just implicit.
    # This means the columns of a are scrambled. We can unscramble them by iterating backward
    # through the pivot elements we recorded and performing the necisarry column swaps.
    for l in range(n-1,-1,-1):
        if indxr[l] != indxc[l]:
            _SWAPCOL(a,indxr[l],indxc[l])
