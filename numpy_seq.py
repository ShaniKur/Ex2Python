import numpy as np

#Q1.1--------------------------------------------------
def half(x):
    return x[::2,1::2]

#Q1.2   n = y.size, m = x.size-------------------------
def outer_product(x,y):
   y = y.reshape(-1,1)
   return x*y

#Q1.3---------------------------------------------------
def extract_logical(x, arr):
    ind = np.array(arr.astype(int) == arr)
    z = np.extract(ind, x)
    return z, ind


# k = length of z
# n = dimension of x
def extract_integer(x, arr):
    z, mask = extract_logical(x, arr)
    k = z.size
    n = x.ndim
    ind = np.arange(n*k).reshape(n,k)
    return z, ind

#Q1.4----------------------------------------------------
def calc_norm(x, axis=0):
       return np.sqrt(np.sum(x**2, axis))
    
def normalization(x, axis=0):
       norm = calc_norm(x,axis)
       if axis == 1:
              norm.reshape(-1,1)
       return x/norm
   
# Q1.5-------------------------------------------------------
def matrix_norm(x, k=1000):
    rand_matrix = np.random.randn(x.shape[0], k) #Draw a n×k matrix of random numbers from the N(0, 1) distribution
    randm_normed =normalization(rand_matrix, axis=0) #Normalize the columns of X
    z= x.dot(randm_normed) #Compute the matrix product Z=A×X
    u=calc_norm(z) #Compute the norm of the columns of Z, denote by u
    return u.max() #The maximal value of u is an approximation of the norm of A

#Q1.6----------------------------------------------------
def det(A):
   sum = 0
   if(A.shape == (2,2)):
         sum = (A[0,0]*A[1,1])-(A[0,1]*A[1,0])
   else:
          for i in range(A.shape[0]):
                 a1 = A[ 1: , :i ]
                 a2 = A[ 1: , i+1: ]
                 newA = np.concatenate( (a1, a2), 1 )
                 sum += (-1)**(i) * A[0,i] * det(newA)
   return sum
    
# Q1.7-------------------------------------------------------
def segment(im, thresh=128):
    if (im > 1).any():
        im_bw = im.mean(axis=2)
        seg = np.where(im_bw < thresh, 0, 255)
    else:
        seg = np.where(im < thresh, 0, 255)
    return seg
    
#Q1.8----------------------------------------------------
def checkInput(x):
    if(x.ndim == 2):
        mask = np.count_nonzero(x, axis=1)
    else:
        mask = np.count_nonzero(x, axis=2)
    mask = mask == 1
    res = np.all(mask) 
    #TD - chek if there are any pairs of 1,1 in the columnes of x (only for the robber)
    return res 
    
    
def linearville(robber, policeman):
    if checkInput(robber) == False:
        print("wrong input!")
        return False
    if checkInput(policeman) == False:
        print("wrong input!")
        return False
    total = robber.shape[0]*robber.shape[1] #depth * rows
    mask = robber == policeman
    checkRows = np.count_nonzero(mask,axis=2)
    num_of_same_rows = np.sum(np.count_nonzero(checkRows, axis=1))
    return (num_of_same_rows == total)

# Q1.9----------------------------------------------------------------
def is_magic(arr):
    main_diagonal_sum = (np.diagonal(arr)).sum()
    antidiagonal_sum= (np.diagonal(np.fliplr(arr))).sum()
    rows_sum= np.sum(arr, axis=1)
    cols_sum = np.sum(arr, axis=0)
    if main_diagonal_sum != antidiagonal_sum :
        return False
    elif not np.all(rows_sum == main_diagonal_sum):
        return False
    elif not np.all(cols_sum == main_diagonal_sum):
        return False
    else:
        return True
