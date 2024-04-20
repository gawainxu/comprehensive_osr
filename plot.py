# visualize the feature ensemble results
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

import matplotlib.pyplot as plt
import numpy as np

values = {"ACCURACY": np.array([90.1, 92.2, 93.1]),
          "AUROC": np.array([69.1, 78.1, 87.3])} 
settings = (r"$\tau$=0.05", r"$\tau$=0.01", r"$\tau$=0.005")

x = np.arange(len(settings))  # the label locations
width = 0.25  
fig, ax1 = plt.subplots()
bottom = np.zeros(3)

metric, v = values.popitem()
p1 = ax1.bar(x, v, width, label=metric, color="blue")
bottom += v
ax1.bar_label(p1, padding=3)
ax1.set_ylabel('AUROC (%)')
ax1.set_ylim(60, 100)

ax2 = ax1.twinx()
metric, v = values.popitem()
p2 = ax2.bar(x+width, v, width, label=metric, color="red")
ax2.bar_label(p2, padding=3)
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(70, 100)


#fig.legend(bbox_to_anchor=(1.2, 1))
fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.75)

ax1.set_title('Values of Inlier Accuracy and Outlier AUROC', fontsize=16)
ax1.set_xticks(x+width/2, settings)
plt.savefig("./test.pdf")
plt.show()