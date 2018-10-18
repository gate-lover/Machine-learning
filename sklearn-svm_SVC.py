import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Preparing two-classed data using random generator
class_1 = 100*np.random.rand(100, 2)
class_2 = -100*np.random.rand(100, 2)

label_1 = np.reshape(np.ones(100, dtype=np.int), 100, 1)
label_2 = np.reshape(-1*np.ones(100, dtype=np.int), 100, 1)

data = np.concatenate((class_1, class_2), axis=0)
label= np.concatenate((label_1, label_2), axis=0)

plt.ion()
fig = plt.figure()
plt.scatter(data[:,0], data[:,1], c=label)


clf = svm.SVC(C=10.0, kernel='linear')
clf.fit(data, label)

plot_decision_regions(data, label, clf=clf)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

