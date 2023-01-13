import numpy as np
import matplotlib.pyplot as plt

with open(f"time/time1.txt") as f0:
    res_file = f0.read().split("\n")[:-1]

TIME1 = 0
for it in range(len(res_file)):
    TIME1 += float(res_file[it].split(" ")[1])
TIME1 /= len(res_file)
# print(TIME1)

with open(f"time/timeRow.txt") as f0:
    res_file = f0.read().split("\n")[:-1]

coresRow, timesRow = np.zeros(len(res_file)), np.zeros(len(res_file))
for it in range(len(res_file)):
    coresRow[it], timesRow[it] = int(res_file[it].split(" ")[0]), float(res_file[it].split(" ")[1])

with open(f"time/timeCol.txt") as f0:
    res_file = f0.read().split("\n")[:-1]

coresCol, timesCol = np.zeros(len(res_file)), np.zeros(len(res_file))
for it in range(len(res_file)):
    coresCol[it], timesCol[it] = int(res_file[it].split(" ")[0]), float(res_file[it].split(" ")[1])

with open(f"time/timeMix.txt") as f0:
    res_file = f0.read().split("\n")[:-1]

coresMix, timesMix = np.zeros(len(res_file)), np.zeros(len(res_file))
for it in range(len(res_file)):
    coresMix[it], timesMix[it] = int(res_file[it].split(" ")[0]), float(res_file[it].split(" ")[1])

ax = plt.subplot()
ax.set_title("Life decompozition")
ax.plot(coresRow, TIME1 / timesRow, "-xr", label="Row")
ax.plot(coresCol, TIME1 / timesCol, "-xg", label="Col")
ax.plot(coresMix, TIME1 / timesMix, "-xb", label="Mix")
ax.set_xlabel("cores")
ax.set_ylabel("boost")

ax.grid()
plt.legend()
plt.show()