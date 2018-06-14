import numpy as np
import tensorflow as tf
def mixup_data(x,y,alpha = 1):
    if alpha >0:
        lam = np.random.beta(alpha,alpha)
    else:
        lam = 1
    print("lam"+str(lam))
    batch_size = len(list(np.array(x,dtype=np.float)))
    print(batch_size)
    index = list(np.arange(batch_size))
    print(index)
    np.random.shuffle(index)
    print(index)
    mixed_x = lam*x + (1-lam)*x[index,:]
    mixed_y = lam*y + (1-lam)*y[index,:]
    return mixed_x,mixed_y

mixup_data(np.array([[1,2,5,4,8],[2,5,5,8,8],[1,2,5,4,8]]),np.array([[1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]]),1)