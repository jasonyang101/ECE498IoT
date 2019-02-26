import matplotlib.pyplot as plt
a = [0.5934,0.4263,0.3860,0.3620,0.3430,0.3294,0.3167,0.3085,0.2993,0.2921]
b = [i for i in range(1,11)]
loss = ['1.6709', '1.6662', '1.6610', '1.6380', '1.6384', '1.6307', '1.6303', '1.6332', '1.6372', '1.6239', '1.6235', '1.6268', '1.6253', '1.6393', '1.5995', '1.5841', '1.5961', '1.5854', '1.5866', '1.5816', '1.5813', '1.5952', '1.5928', '1.5690', '1.5737']
tf_a = [float(loss[i]) for i in range(len(loss))]
tf_b = [i for i in range(1,26)]
# plt.plot(b,a)
# plt.title("Part 1 Keras")
# plt.ylabel("loss values")
# plt.xlabel("epoch")
# plt.savefig("Part1Keras.png")
# plt.show()

plt.plot(b,a)
plt.title("Part 2 Tensorflow")
plt.ylabel("loss values")
plt.xlabel("epoch")
plt.savefig("Part2TF.png")
plt.show()
