# From calculation, it is expected that the local minimum occurs at x=9/4. The function is f(x)=x^4−3x^3+2
import time
import numpy as np
import matplotlib.pyplot as plt

init = 0.6
tic = 0.005
cur_x = init # The algorithm starts at x=6
gamma = 0.01 # step size multiplier
precision = 0.0001
previous_step_size = cur_x
n_iter = 100

def f(x):
	#return x**4 + 3 * x**2 
	#return x**4 - 2 * x**2 + x + 2.1
	#return x**2
    #return x ** 3
    return 1/2*(x-2)**2 + 1/2*(x+1)**2
def df(x):
    #return 4 * x**3 + 6 * x
    #return 4 * (x**3) - 4 * x + 1
    #return 2 * x
    #return 3 * (x**2)
    return 2*(x-2) + 2*(x+1)

def valB(x,const):
	return const - df(x)*x

def sign (x):
    if x < 0 :
       return 1
    else :
       return - 1
start1 = time.time()

tab = []
cur_x = init
i=0
n = 0
tab1 = []
while n < n_iter : #(time.time() - start1) <= tic : previous_step_size > precision: #
    prev_x = cur_x
    cur_x += -gamma * df(prev_x)
    #print("curV1 =  %f" % (cur_x))
    previous_step_size = abs(cur_x - prev_x)
    i= i + 1
    n = n + 1
    tab1.append(cur_x)

end = time.time()
tab.append(tab1)
print("1- Excecution time = %f" %(end - start1))
print("1- The local minimum occurs at %f" % cur_x)
print("1- iteration = %f" % i)



#part 2
tab1 = []
p_v = 0
cur_x = init
previous_step_size = cur_x
start = time.time()
i=0
n = 0
while n < n_iter : #(time.time() - start) <= tic: #
    prev_x = cur_x
    cur = df(prev_x)
    #new_c = cur_x + (-gamma * ( df(p_v) + df(prev_x))) 
    if p_v * cur > 0   :
        #print("grad = {}".format(cur))
        #print("p_v = {}".format(p_v))
        cur_x += -gamma * (p_v+ cur)
        #print(cur_x)
    else :
        #print(cur_x)
        cur_x += -gamma * cur
    #cur_x += -gamma * (max(df(p_v),df(prev_x) )+ df(prev_x))
    p_v = cur
    #print("curV1 =  %f" % (cur_x))
    previous_step_size = abs(cur_x - prev_x)
    i = i+1
    n = n + 1
    tab1.append(cur_x)

end = time.time()
tab.append(tab1)
print("2- Excecution time = %f" %(end - start))
print("2- The local minimum occurs at %f" % cur_x)
print("2- iteration = %f" % i)

#part 4
tab1=[]
cur_x = init
previous_step_size = cur_x
start1 = time.time()
prev_df = 0
beta1 = 0.9
beta2 = 0.999
m = 0
r = 0
t = 1
eps = 1e-8
i=0
n = 0
while n < n_iter : #(time.time() - start1) <= tic : #
    prev_x = cur_x
    m = beta1 * m + (1 - beta1) * df(prev_x)
    r = beta2 * r + (1 - beta2) * (df(prev_x) ** 2)
    mhat = m / (1 - beta1 ** t)
    rhat = np.sqrt(r / (1 - beta2 ** t) + eps)
    cur_x += -(gamma * mhat/rhat)
    p_v = prev_x
    #print("curX =  %f " % (cur_x))
    #print("f(curX) =  %f " % (f(cur_x)))
    previous_step_size = abs(cur_x - prev_x)
    t = t+1
    i=i+1
    n = n + 1
    tab1.append(cur_x)

end = time.time()
tab.append(tab1)
print("4- Excecution time = %f" %(end - start1))
print("4- The local minimum occurs at %f" % cur_x)
print("4- iteration = %f" % i)

#part 3
tab1=[]
cur_x = init
previous_step_size = cur_x
start = time.time()
prev_df = 0
beta1 = 0.9
beta2 = 0.999
m = 0
r = 0
t = 1
eps = 1e-8
p_v1 = 0
p_v2 = 0
prev_x2 = 0
i=0
mhat=0
n = 0
k = 0
while n < n_iter : #(time.time() - start) <= tic: #
    prev_x = cur_x
    cur = df(prev_x)
    alp = m
    #p_v1 = m
    #tempm = (beta1 * m + (1 - beta1) * (df(prev_x2)+ df(prev_x)))/ (1 - beta1 ** t)
    #tempr = np.sqrt((beta2 * r + (1 - beta2) * (df(prev_x) ** 2))/(1 - beta2 ** t) + eps)
    #new_c = prev_x + (-(gamma * tempm/tempr))
    if p_v1 * cur  > 0 :
        m = beta1 * m + (1 - beta1) * (p_v1+ cur)
        #r = beta2 * r + (1 - beta2) * ((df(prev_x2) + df(prev_x)) ** 2)
        #print("new_m = {}".format(beta1 * m + (1 - beta1) * (df(prev_x2)+ df(prev_x)) ))
        #print("sign = {}".format(df(cur_x)))
        #print(m) 
    else :
        m = beta1 * m + (1 - beta1) * cur
    #m = beta1 * m + (1 - beta1) * (max(df(prev_x2),df(prev_x) )+ df(prev_x))
    r = beta2 * r + (1 - beta2) * (df(prev_x) ** 2)
    #print(m)
    mhat = m / (1 - beta1 ** t)
    rhat = np.sqrt(r / (1 - beta2 ** t) + eps)
    cur_x += -(gamma * mhat/rhat)
    #print("curX =  %f " % (cur_x))
    #print("f(curX) =  %f " % (f(cur_x)))
    previous_step_size = abs(cur_x - prev_x)
    p_v1 = cur
    t = t+1
    i = i + 1
    n = n + 1
    tab1.append(cur_x)

end = time.time()
tab.append(tab1)
print("3- Excecution time = %f" %(end - start))
print("3- The local minimum occurs at %f" % cur_x)
print("3- iteration = %f" % i)


# When a function's slope is zero at x, and the second derivative at x is:

    #less than 0, it is a local maximum
    #greater than 0, it is a local minimum
	#equal to 0, then the test fails (there may be other ways of finding out though)
i=0
n = 0
tab1 = []
cur_x = init
new_a = 0
start1 = time.time()
while n < n_iter : #(time.time() - start1) <= tic : #previous_step_size > precision:
    prev_x = cur_x
    new_a = new_a + (df(prev_x))**2
    cur_x += - (gamma * df(prev_x)/(np.sqrt(new_a) + eps))
    #print("curV1 =  %f" % (cur_x))
    previous_step_size = abs(cur_x - prev_x)
    i= i + 1
    n = n + 1
    tab1.append(cur_x)

end = time.time()
tab.append(tab1)
print("5- Excecution time = %f" %(end - start1))
print("5- The local minimum occurs at %f" % cur_x)
print("5- iteration = %f" % i)

i=0
n = 0
tab1 = []
cur_x = init
new_a = 0
p_v = 0
start = time.time()
while n < n_iter : #(time.time() - start) <= tic: #
    prev_x = cur_x
    new_a = new_a + (df(prev_x))**2
    cur = df(prev_x)
    if cur * p_v > 0  :
        cur_x += - (gamma * (p_v+ cur)/(np.sqrt(new_a) + eps))
    else :
        cur_x += - (gamma * cur/(np.sqrt(new_a) + eps))
    #print("curV1 =  %f" % (cur_x)) 
    #cur_x += - (gamma * (max(df(p_v),df(prev_x) )+ df(prev_x))/(np.sqrt(new_a) + eps))
    previous_step_size = abs(cur_x - prev_x)
    i= i + 1
    n = n + 1
    p_v = cur
    tab1.append(cur_x)

end = time.time()
tab.append(tab1)
print("6- Excecution time = %f" %(end - start))
print("6- The local minimum occurs at %f" % cur_x)
print("6- iteration = %f" % i)


b = range(n_iter)
plt.subplots(1, 1)
labels = ["SGD","ASGD","Adam","AAdam","Adagrad","AAdagrad"]
for g in range(len(tab)):
   plt.plot(tab[g],label=labels[g])
plt.legend(loc='upper right')
plt.title(r'$f(x) = \frac{1}{2}(x-2)^2 + \frac{1}{2}(x+1)^2 $, starting point $x = 0.6$. ')
plt.xlabel("iterations")
plt.ylabel("values of x")
plt.show()
