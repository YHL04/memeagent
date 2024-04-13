

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns


dir = os.listdir("logs")
dir.sort()
latestfile = "logs/" + dir[-1]

data = pd.read_csv(latestfile,
                   names=["Time",
                          "Updates",
                          "Frames",
                          "Loss",
                          "PLoss",
                          "IntrLoss",
                          "Reward",
                          "Intrinsic",
                          "Epsilon",
                          "Arm",
                          "ReplayRatio"]
                   )

data["Temp"] = np.arange(len(data["Reward"])) // 50
sns.lineplot(data=data, x="Temp", y="Reward")
plt.show()

plt.subplot(8, 1, 1)
plt.title("Time (Sec)")
plt.plot(data["Time"])

plt.subplot(8, 1, 2)
plt.title("Updates")
plt.plot(data["Updates"])

plt.subplot(8, 1, 3)
plt.title("Frames")
plt.plot(data["Frames"])

plt.subplot(8, 1, 4)
plt.title("Loss (Log scale)")
plt.yscale("log")
plt.plot(data["Loss"])

plt.subplot(8, 1, 5)
plt.title("Policy Loss (Log scale)")
plt.yscale("log")
plt.plot(data["PLoss"])

plt.subplot(8, 1, 6)
plt.title("IntrLoss (Log scale)")
plt.yscale("log")
plt.plot(data["IntrLoss"])

plt.subplot(8, 1, 7)
plt.title("Reward")
plt.plot(data["Reward"])
plt.plot(data["Reward"].rolling(200).mean())

plt.subplot(8, 1, 8)
plt.title("Intrinsic")
plt.plot(data["Intrinsic"])
plt.plot(data["Intrinsic"].rolling(200).mean())


plt.show()
