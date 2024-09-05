# visualize the feature ensemble results
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

import matplotlib.pyplot as plt
import numpy as np

"""
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
"""


species = ["SimCLR ResNet18 @1", "SimCLR ResNet34 @1", "SimCLR ResNet50 @1", "MoCo ResNet18 @1", "MoCo ResNet34 @1",
           "SimCLR ResNet18 @5", "SimCLR ResNet34 @5", "SimCLR ResNet50 @5", "MoCo ResNet18 @5", "MoCo ResNet34 @5"]
accs = {"Plain": (40.5, 42.07, 45.54, 35.17, 38.21, 67.29, 68.62, 71.31, 60.79, 63.96),
        "CutMix": (41.01, 42.45, 46.68, 34.76, 37.63 ,68.13, 68.99, 72.25, 60.72, 64.11),
        "GradMix": (41.63, 42.49, 47.02, 42.49, 39.41,68.31, 69.04, 72.33, 64.66, 65.2)}


x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_figwidth(15)

colors1 = ["#6890F0", "#6890F0", "#6890F0", "#6890F0", "#6890F0", "#6890F0", "#6890F0", "#6890F0", "#6890F0", "#6890F0"]
colors2 = ["gold", "gold", "gold", "gold", "gold", "gold", "gold", "gold", "gold", "gold"]
colors3 = ["#1b9e77", "#1b9e77", "#1b9e77", "#1b9e77", "#1b9e77", "#1b9e77", "#1b9e77", "#1b9e77", "#1b9e77", "#1b9e77"] 
colors = [colors1, colors2, colors3]

for i, (attribute, measurement) in enumerate(accs.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])
    ax.bar_label(rects, padding=3, fontsize=6)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)', fontsize=15)
ax.set_xticks(x + width, species, fontsize=8)
ax.legend(loc='upper left', ncol=3)
ax.set_ylim(0, 80)

plt.show()
fig.savefig("./plots/ssl_acc.pdf")


"""
plain_acc1 = [38.34, 40.43, 42.28, 43]
plain_acc5 = [64.88, 67.26, 69.1, 69.43]

cut_acc1 = [39.45, 40.86, 42.8, 43.38]
cut_acc5 = [67.37, 68.48, 70.45, 70.46]

grad_acc1 = [38.71,	41.61, 43.58, 44.68]
grad_acc5 = [65.67, 68.41, 70.19, 71.63]


epochs = [100, 200, 300, 400]

plt.plot(epochs, plain_acc1, label="Plain @1", linewidth=2.5)
plt.plot(epochs, cut_acc1, label="CutMix @1", linewidth=2.5)
plt.plot(epochs, grad_acc1, label="GradMix @1", linewidth=2.5)


plt.plot(epochs, plain_acc5, label="Plain @5", linewidth=2.5)
plt.plot(epochs, cut_acc5, label="CutMix @5", linewidth=2.5)
plt.plot(epochs, grad_acc5, label="GradMix @5", linewidth=2.5)

plt.ylabel('Accuracy (%)', fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.legend()
plt.savefig("./plots/ssl_acc_change.pdf")
"""

"""
# corruption classification, barplot


grad_acc = [42.9, 17.22, 13.24, 27.1, 26.02, 34.94, 13.56,
            14.42, 16.2, 36.18, 19.06, 24.86, 17.3, 30.34, 15.3, 23.24] #[89.66, 88.42, 89.42, 84.5 ,80.7, 75.16, 51.08 ,56.54, 61.7, 79.04 ,80.92 ,73.92 ,59.86 ,77.82 ,87.18, 75.73]
plain_acc = [42.22, 17.32, 12.38, 25.7, 25.14, 35.12, 13.98,
            13.24, 17.86, 35.16, 18.48 ,25.52, 17.48, 29.86, 15.32, 22.99] #[89.64, 82.78, 77.46, 80, 77.7, 73.36, 51.84, 55.46, 60.54, 81.04, 71.86, 72.06, 62.76, 77.94, 73.3, 72.52]

grad_clean = 60.6 #91.7
plain_clean = 61.8 #90.6

species = ["Brightness",
"Contrast",
"Defocus Blur",
"Elastic Transform",
"Fog",
"Frost",
"Gaussian Noise",
"Glass Blur",
"Impulse Noise",
"JPEG",
"Motion Blur",
"Pixelate",
"Shot Noise",
"Snow",
"Zoom Blur",
"Average"]


grad_acc_dis = [grad_clean - i for i in grad_acc]
plain_acc_dis = [plain_clean - i for i in plain_acc]

data_dict = {"GradMix": grad_acc_dis, "Plain": plain_acc_dis}

fig, ax = plt.subplots(layout='constrained')
fig.set_figwidth(15)

x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

for attribute, measurement in data_dict.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, fontsize=6)
    multiplier += 1
    
    
ax.set_ylabel('Accuracy Drop (%)', fontsize=15)
ax.set_xticks(x + width, species, fontsize=8)
ax.legend(loc='upper left', ncol=2)

ax.set_title("Accuracy Drop with Corrupted Images (TinyImgNet)", fontsize=16)
fig.savefig("./plots/corruption_tinyimgnet.pdf")
"""

"""
# corruption classification, barplot
# https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python


grad_acc = [42.9, 17.22, 13.24, 27.1, 26.02, 34.94, 13.56,
            14.42, 16.2, 36.18, 19.06, 24.86, 17.3, 30.34, 15.3, 23.24] #[89.66, 88.42, 89.42, 84.5 ,80.7, 75.16, 51.08 ,56.54, 61.7, 79.04 ,80.92 ,73.92 ,59.86 ,77.82 ,87.18, 75.73]
plain_acc = [42.22, 17.32, 12.38, 25.7, 25.14, 35.12, 13.98,
            13.24, 17.86, 35.16, 18.48 ,25.52, 17.48, 29.86, 15.32, 22.99] #[89.64, 82.78, 77.46, 80, 77.7, 73.36, 51.84, 55.46, 60.54, 81.04, 71.86, 72.06, 62.76, 77.94, 73.3, 72.52]

grad_clean = 60.6 #91.7
plain_clean = 61.8 #90.6

species = ["Brightness",
"Contrast",
"Defocus Blur",
"Elastic Transform",
"Fog",
"Frost",
"Gaussian Noise",
"Glass Blur",
"Impulse Noise",
"JPEG",
"Motion Blur",
"Pixelate",
"Shot Noise",
"Snow",
"Zoom Blur",
"Average",
]



grad_acc_dis = [grad_clean - i for i in grad_acc]
plain_acc_dis = [plain_clean - i for i in plain_acc]

data_dict = {"GradMix": grad_acc_dis, "Plain": plain_acc_dis}


angles = np.linspace(0, 2*np.pi, len(species), endpoint=False)
stats_plain = np.concatenate((plain_acc_dis, [grad_acc_dis[0]]))
stats_grad = np.concatenate((grad_acc_dis, [grad_acc_dis[0]]))
angles = np.concatenate((angles, [angles[0]]))

species = species + ["Brightness"]
species = np.array(species)

fig = plt.figure()
fig.set_figwidth(11)
ax = fig.add_subplot(121, polar=True)
ax.set_ylim(10, 50)
ax.plot(angles, stats_plain, "o-", linewidth=2, label="Vanilia")
ax.fill(angles, stats_plain, alpha=0.25)
ax.plot(angles, stats_grad, "o-", linewidth=2, label="GradMix")
ax.fill(angles, stats_grad, alpha=0.25)
ax.set_thetagrids(angles*180/np.pi, species)
ax.set_title("tinyImgNet")


grad_acc_cifar = [89.66, 88.42, 89.42, 84.5 ,80.7, 75.16, 51.08 ,56.54, 61.7, 79.04 ,80.92 ,73.92 ,59.86 ,77.82 ,87.18, 75.73]
plain_acc_cifar = [89.64, 82.78, 77.46, 80, 77.7, 73.36, 51.84, 55.46, 60.54, 81.04, 71.86, 72.06, 62.76, 77.94, 73.3, 72.52]

grad_clean_cifar = 91.7
plain_clean_cifar = 90.6

grad_acc_dis_cifar = [grad_clean_cifar - i for i in grad_acc_cifar]
plain_acc_dis_cifar = [plain_clean_cifar - i for i in plain_acc_cifar]

data_dict_cifar = {"GradMix": grad_acc_dis_cifar, "Plain": plain_acc_dis_cifar}

stats_plain_cifar = np.concatenate((plain_acc_dis_cifar, [grad_acc_dis_cifar[0]]))
stats_grad_cifar = np.concatenate((grad_acc_dis_cifar, [grad_acc_dis_cifar[0]]))


ax = fig.add_subplot(122, polar=True)
ax.set_ylim(0, 45)
ax.plot(angles, stats_plain_cifar, "o-", linewidth=2)
ax.fill(angles, stats_plain_cifar, alpha=0.25)
ax.plot(angles, stats_grad_cifar, "o-", linewidth=2)
ax.fill(angles, stats_grad_cifar, alpha=0.25)
ax.set_thetagrids(angles*180/np.pi, species)
ax.set_title("Cifar10")

legendEntries = ("Vanilia","GradMix")
fig.legend(loc="upper center")
plt.show()
fig.savefig("./plots/corruption_radar.pdf")
"""