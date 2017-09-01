#!/usr/bin/python3

import collections
import numpy as np

def calculate_hessenberg(A):
    m = A.shape[0] # m --> rows
    n = A.H.shape[0] # n --> cols
    H = np.matrix(A)
    for j in range(0, m-2):
        aux = range(j+1,m)
        x = H[np.ix_(aux,[j])]
        x[0] = x[0] + np.sign(x[0]) * np.linalg.norm(x)
        n = np.linalg.norm(x)
        if n > 0:
            u = x/np.linalg.norm(x)
            aux2 = range(j,m)
            Hh = H[np.ix_(aux,aux2)] - 2*u*(u.H * H[np.ix_(aux, aux2)])
            for r in aux:
                for c in aux2:
                    H[r,c] = Hh[r-j-1,c-j]
            aux3 = range(0,m)
            Hh = H[np.ix_(aux3,aux)] - 2 * (H[np.ix_(aux3,aux)] * u) * u.T
            for r in aux3:
                for c in aux:
                    H[r,c] = Hh[r,c-1-j]
        else:
            u = x;
    return H

def diag(n):
    D = np.zeros((n,n))
    for i in range(0,n):
        D[i,i] = 1
    return D


def calculate_QR(A):
    n = A.shape[0] # n --> rows
    m = A.H.shape[0] # m --> cols
    R = np.matrix(A)
    Q = np.matrix(diag(n))
    for i in range(0,n):
        for j in range(i+1,n):
            if R[j,i] == 0:
                c = 1
                s = 0
            elif np.absolute(R[j,i]) < np.absolute(R[i,i]):
                t = R[j,i] / R[i,i]
                c = 1 / np.sqrt(1+t**2)
                s = c*t
            else:
                z = R[i,i] / R[j,i]
                s = 1 / np.sqrt(1+z**2)
                c = s*z
            G = np.matrix([[c,s],[-1*s,c]])
            aux = range(i,n)
            Rr = G * R[np.ix_([i,j],aux)]
            for c in aux:
                R[i,c] = Rr[0,c-i]
                R[j,c] = Rr[1,c-i]

            aux2 = range(0,n)
            Qq = G * Q[np.ix_([i,j],aux2)]
            for c2 in aux2:
                Q[i,c2] = Qq[0,c2]
                Q[j,c2] = Qq[1,c2]
    return (Q.H,R)

def main():
    A = np.matrix('1.,2.,3.,4.;-1.,-2.,-3.,-4.;5,6,7,8;9,0,1,2')
    #print(calculate_hessenberg(A))
    (Q,R) = calculate_QR(A)
    print(Q)
    print(R)

if __name__ == "__main__":
    main()
