import numpy as np

# t1 = np.array([4, 5, 6])
# print(t1.dtype)
# print(t1.shape)
# print(t1[0])

# t1 = np.array([[4, 5, 6], [1, 2, 3]])
# print(t1.dtype)
# print(t1.shape)
# print(t1[0][0], t1[0][1], t1[1][1])

a = np.zeros((3, 3), dtype = int)
b = np.ones((4, 5), dtype = int)
c = np.identity(4, dtype = int)
d = np.random.randint(1, 10, size = (3, 2))
e = np.random.rand(2, 3)
f = np.random.randn(3, 3)
g=np.random.normal(10,10,(5,3))

Z = np.arange(10, 50)
print(Z)
Z = np.arange(9).reshape(3,3)
print(Z)
Z = np.eye(5, k = 2)
print(Z)


# print(c)
# print(d)
# print(e)
# print(f)
# print(g)
# print(np.random.choice(10))