import matplotlib.pyplot as plt
a = [0.5934,0.4263,0.3860,0.3620,0.3430,0.3294,0.3167,0.3085,0.2993,0.2921]
b = [i for i in range(1,11)]
loss = ['1.8322', '1.8090', '1.7274', '1.6291', '1.6057', '1.6246', '1.6181', '1.6037', '1.6051', '1.6086']
tf_a = [float(loss[i]) for i in range(len(loss))]
tf_b = [i for i in range(1,11)]
# plt.plot(b,a)
# plt.title("Part 1 Keras")
# plt.ylabel("loss values")
# plt.xlabel("epoch")
# plt.savefig("Part1Keras.png")
# plt.show()

plt.plot(tf_b,tf_a)
plt.title("Part 2 Tensorflow")
plt.ylabel("loss values")
plt.xlabel("epoch")
plt.savefig("Part2TF.png")
plt.show()
