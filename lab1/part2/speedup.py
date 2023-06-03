import matplotlib.pyplot as plt
x = [2, 3, 4]
y = [1.95, 1.63, 2.38]
plt.plot(x, y, 's-', color='r')
plt.xlabel("num of thread")
plt.ylabel("speed up")
plt.show()
