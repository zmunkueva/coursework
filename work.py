import numpy as np
import copy
data = open("дата\\cond_calibrate.dat").read().split('\n')
freq_ch = np.array([data[k].split('  ')[4] for k in range(1, len(data) - 1)])
cp_ch = np.array([data[k].split('  ')[1] for k in range(1, len(data) - 1)])
tm_ch = np.array([data[k].split('  ')[0] for k in range(1, len(data) - 1)])

freq = np.array([float(c) for c in freq_ch])
cp = np.array([float(c) for c in cp_ch])
tm = np.array([float(c) for c in tm_ch])
#dict.setdefault(key[, default]) возвращение значения
dict = {0: 25, 1: 120, 2: 500, 3: 1000, 4: 5000, 5: 10000, 6: 20000, 7: 50000, 8: 100000,
9: 200000, 10: 500000, 11: 1000000}

key = 11

x = np.array([freq, cp, tm]).T  #матрица - оригинал
value = dict.setdefault(key)
print('key value: ',value, '\n')

matrix = x[ x[:,0] == value ]
print('numbers of value: ', len(matrix))


#matrix = np.delete(matrix, 0 , axis=0)
#matrix = np.delete(matrix, 0 , axis=0)


import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

def f(x, a0, a1, a2, a3, a4):#, a5, a6, a7, a8):
    return a0 + a1*x + a2*x**2 + a3*x**3  + a4*x**4 #+  a5*x**5 + a6*x**6 + a7*x**7 + a8*x**8

plt.scatter(matrix[:,2], matrix[:,1])

koef,q=curve_fit(f,matrix[:,2],matrix[:,1])
a0, a1, a2, a3, a4= koef #,a2, a3 , a4, a5, a6, a7, a8

#print(a0, a1, a2, a3, a4, a5, a6, a7, a8)
aproximated_data = f(matrix[:,2], a0, a1, a2, a3, a4)#, a5, a6, a7, a8


UPPER_BOUND = 0.00086
LOWER_BOUND = 0.00086


plt.plot(matrix[:,2], aproximated_data + UPPER_BOUND,'g')
plt.plot(matrix[:,2], aproximated_data,'r')
plt.plot(matrix[:,2], aproximated_data - LOWER_BOUND,'g')

in_upper = matrix[:,1] < aproximated_data + UPPER_BOUND
in_lower = matrix[:,1] > aproximated_data - LOWER_BOUND

matrix_cleared = np.asarray([[matrix[i,2], matrix[i,1]] if (in_upper[i] & in_lower[i]) else [matrix[i,2], aproximated_data[i]] for i in range(len(matrix))])
plt.scatter(matrix[:,2], matrix_cleared[:,1])

mean_value = np.mean(matrix_cleared[:,1])
#print(str(10.0))
print('mean value: ', mean_value,'\n')
plt.plot(matrix[:,2], [mean_value for l in matrix[:,2]], 'purple')

plt.show()

print('appropriate mean value?(y/n): ')
ans = input()
if ans == 'y':
    with open('средняя ёмкость.txt', 'a', encoding = 'cp1252') as file:
        file.write(str(mean_value) + '\n')
