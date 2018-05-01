import numpy as np
import matplotlib.pyplot as plt

N=5
lb,ub = -6,6
freq_cos = 0.05
freq_sin = 0.3
f = lambda x: np.cos(freq_cos*2*np.pi*x)*np.sin(freq_sin*2*np.pi*x)
x = np.linspace(lb,ub,N)
print(x)
print(f(x))

p_f, = plt.plot(x,f(x))
plt.legend([p_f], ['function'])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
