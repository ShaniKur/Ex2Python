import numpy as np

#Q1.1--------------------------------------------------
def half(x):
    return x[::2,1::2]

#Q1.2   n = y.size, m = x.size-------------------------
def outer_product(x,y):
   y = y.reshape(-1,1)
   return x*y

x = np.array([1,2,3,4,5])
y = np.array([10,20,30])
print("XY= ", outer_product(x,y))

#Q1.3---------------------------------------------------
def extract_logical(x, arr):
    ind = np.array(arr.astype(int) == arr)
    print(ind)

    z = np.extract(ind, x)
    print(z)

    return z, ind


# k = length of z
# n = dimension of x
def extract_integer(x, arr):
    z, mask = extract_logical(x, arr)
    indices = np.arange(len(arr)**arr.ndim).reshape(arr.shape)
    print(indices)
    k = z.size
    n = x.ndim

    ind = np.zeros(n*k).reshape(k,n)
    print(ind)



arr = np.array([[1.2, 1.3, 9],
                [1.2, 1.3, 9],
                [5, 1.3, 9]])

print(extract_integer(arr, arr))

#Q1.4----------------------------------------------------
def calc_norm(x, axis=0):
       return np.sqrt(np.sum(x**2, axis))
    
def normalization(x, axis=0):
       norm = calc_norm(x,axis)
       if axis == 1:
              norm.reshape(-1,1)
       return x/norm
   

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
   

#       np.zeros(depth, rows, cols)
robber = np.zeros((2, 2, 3),np.int32)
# 2 days
# 3 stores
# 2 hours
# first day - first hour
robber[0][0][0] = 1
robber[0][0][1] = 0
robber[0][0][2] = 0
# second hour
robber[0][1][0] = 0
robber[0][1][1] = 1
robber[0][1][2] = 0

# second day - first hour
robber[1][0][0] = 1
robber[1][0][1] = 0
robber[1][0][2] = 0
# second hour
robber[1][1][0] = 0
robber[1][1][1] = 1
robber[1][1][2] = 0

###########

policeman = np.zeros((2, 3)).astype(int)
# 3 stores
# 2 hours
# first hour
policeman[0][0] = 0
policeman[0][1] = 1
policeman[0][2] = 0
# second hour
policeman[1][0] = 0
policeman[1][1] = 0
policeman[1][2] = 1

check = np.ones((2), dtype=np.int32)
print(linearville(robber,policeman))
