import numpy as np
def ADD(x,y,c):
    if (x==0 and y==0 and c==0):
        S=str(0)
        c=0
    elif (x==0 and y==0 and c==1):
        S=str(1)
        c=0
    elif (x==0 and y==1 and c==0):
        S=str(1)
        c=0
    elif (x==0 and y==1 and c==1):
        S=str(0)
        c=1
    elif (x==1 and y==0 and c==0):
        S=str(1)
        c=0
    elif (x==1 and y==0 and c==1):
        S=str(0)
        c=1
    elif (x==1 and y==1 and c==0):
        S=str(0)
        c=1
    else:
        S=str(1)
        c=1
    return S,c
def AMA1(x,y,c):
    if (x==0 and y==0 and c==0):
        S=str(1)
        c=0
    elif (x==0 and y==0 and c==1):
        S=str(1)
        c=0
    elif (x==0 and y==1 and c==0):
        S=str(0)
        c=1
    elif (x==0 and y==1 and c==1):
        S=str(0)
        c=1
    elif (x==1 and y==0 and c==0):
        S=str(1)
        c=0
    elif (x==1 and y==0 and c==1):
        S=str(0)
        c=1
    elif (x==1 and y==1 and c==0):
        S=str(0)
        c=1
    else:
        S=str(0)
        c=1
    return S,c

def AMA2(x,y,c):
    if (x==0 and y==0 and c==0):
        S=str(0)
        c=0
    elif (x==0 and y==0 and c==1):
        S=str(1)
        c=0
    elif (x==0 and y==1 and c==0):
        S=str(0)
        c=0
    elif (x==0 and y==1 and c==1):
        S=str(1)
        c=0
    elif (x==1 and y==0 and c==0):
        S=str(0)
        c=1
    elif (x==1 and y==0 and c==1):
        S=str(0)
        c=1
    elif (x==1 and y==1 and c==0):
        S=str(0)
        c=1
    else:
        S=str(1)
        c=1
    return S,c
def AMA3(x,y,c):
    if (x==0 and y==0 and c==0):
        S=str(0)
        c=0
    elif (x==0 and y==0 and c==1):
        S=str(0)
        c=0
    elif (x==0 and y==1 and c==0):
        S=str(1)
        c=0
    elif (x==0 and y==1 and c==1):
        S=str(1)
        c=0
    elif (x==1 and y==0 and c==0):
        S=str(0)
        c=1
    elif (x==1 and y==0 and c==1):
        S=str(0)
        c=1
    elif (x==1 and y==1 and c==0):
        S=str(1)
        c=1
    else:
        S=str(1)
        c=1
    return S,c
def LOA(x,y,c):
    if (x==0 and y==0 and c==0):
        S=str(0)
        c=0
    elif (x==0 and y==0 and c==1):
        S=str(0)
        c=0
    elif (x==0 and y==1 and c==0):
        S=str(1)
        c=0
    elif (x==0 and y==1 and c==1):
        S=str(1)
        c=0
    elif (x==1 and y==0 and c==0):
        S=str(1)
        c=0
    elif (x==1 and y==0 and c==1):
        S=str(1)
        c=0
    elif (x==1 and y==1 and c==0):
        S=str(1)
        c=1
    else:
        S=str(1)
        c=1
    return S,c

N = 14  # 加法器总位宽
k = 1  # 低位近似部分位宽
I = 14  # 整数部分位宽
print(N,k,I)


def approadd(x, y):
    i = 0
    x_n = y_n = 0
    X_comple = Y_comple = ""
    S = S_comple = ""
    s = 0.0

    if (x < 0):
        x = -x
        x_n = 1
    if (y < 0):
        y = -y
        y_n = 1

    if int(x * (2 ** (N - I))) == 0:
        x_n = 0
    if int(y * (2 ** (N - I))) == 0:
        y_n = 0

    X = bin(int(x * (2 ** (N - I))))[2:]
    Y = bin(int(y * (2 ** (N - I))))[2:]
    X_len = len(X)
    Y_len = len(Y)

    if (X_len > N or Y_len > N):
        #  print("x or y is too large")
        #  print('a:')
        #  print(x*((-1)**(x_n)))
        #  print('b:')
        #  print(y*((-1)**(y_n)))
        while i < (N - X_len + 1):
            X = str(0) + X
            i = i + 1
        i = 0
        while i < (N - Y_len + 1):
            Y = str(0) + Y
            i = i + 1
        i = 0
    else:
        while i < (N - X_len + 1):
            X = str(0) + X
            i = i + 1
        i = 0
        while i < (N - Y_len + 1):
            Y = str(0) + Y
            i = i + 1
        i = 0

    if x_n == 1:
        c = 1
        while i < (N + 1):
            X_comple = str((1 - int(X[N - i])) ^ c) + X_comple
            c = (1 - int(X[N - i])) & c
            i = i + 1
        X = str(x_n) + X_comple
        i = 0
    else:
        X = str(x_n) + X
    if y_n == 1:
        c = 1
        while i < (N + 1):
            Y_comple = str((1 - int(Y[N - i])) ^ c) + Y_comple
            c = (1 - int(Y[N - i])) & c
            i = i + 1
        Y = str(y_n) + Y_comple
        i = 0
    else:
        Y = str(y_n) + Y

    c = 0
    while i < k:
        S_tmp, c = AMA3(int(X[N + 2 - 1 - i]), int(Y[N + 2 - 1 - i]), c)
        S = S_tmp + S
        i = i + 1
    while i < N + 2:
        S_tmp, c = ADD(int(X[N + 2 - 1 - i]), int(Y[N + 2 - 1 - i]), c)
        S = S_tmp + S
        i = i + 1
    i = 0

    if int(S[0]) == 1:
        c = 1
        while i < N + 1:
            S_comple = str((1 - int(S[N + 1 - i])) ^ c) + S_comple
            c = (1 - int(S[N + 1 - i])) & c
            i = i + 1
        i = 0
        S = S[0] + S_comple

    while i < N + 1:
        s = s + int(S[N + 2 - 1 - i]) * (2 ** (i))
        i = i + 1
    i = 0
    s = (s / (2 ** (N - I))) * ((-1) ** (int(S[0])))

    return s

x=approadd(81,-4)
print(x)