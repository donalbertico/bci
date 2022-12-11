import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string, random

lit = pd.read_excel('literature_review.xlsx',2)

fig, ax = plt.subplots()

features2 = lit['features1']
features2 = np.array([i for i in features2 if i == i])
features3 = lit['features3']
features3 = np.array([i for i in features3 if i == i])
features1 = lit['features'].values
features1 = np.array([i for i in features1 if i == i])

features = np.concatenate((features1,features2,features3))
unique, count = np.unique(features, return_counts=True)
features = pd.DataFrame({
        'feature' : unique,
        'id' : [np.where(unique == i)[0][0] for i in unique],
        'count' : count,
    })
features = features.sort_values('count', ascending=True)


method = lit['method']
method = np.array([i for i in method if i == i])
unique, count = np.unique(method, return_counts= True)
methods = pd.DataFrame({
        'feature' : unique,
        'id' : [np.where(unique == i)[0][0] for i in unique],
        'count' : count,
    })
methods = methods.sort_values('count', ascending=True)


classifier = lit['classifier']
classifier = np.array([i for i in classifier if i == i])
classifier1 = lit['classifier1']
classifier1 = np.array([i for i in classifier1 if i == i])
classifier = np.concatenate((classifier,classifier1))
unique, count = np.unique(classifier, return_counts= True)
classifiers = pd.DataFrame({
        'feature' : unique,
        'id' : [np.where(unique == i)[0][0] for i in unique],
        'count' : count,
    })
classifiers = classifiers.sort_values('count', ascending=False)

classifiers.plot.bar(x= 'id', y= 'count', legend= False, ax = ax, title='Classifier', color = 'blue', width = 0.8)
x_legend = '\n'.join(f'{n} - {name}' for n,name in zip(range(len(unique)), unique))
t = ax.text(.7,.1,x_legend, transform=ax.figure.transFigure)
fig.subplots_adjust(right=.60)
plt.ylim(0,14)
plt.xlim(0,30)
ax.set_aspect(2)

methods.plot.barh(x= 'feature', y= 'count', legend= False, ax = ax, title='Method', color = 'black', width = 0.8)
x_legend = '\n'.join(f'{n} - {name}' for n,name in zip(range(len(unique)), unique))
# t = ax.text(.7,.1,x_legend, transform=ax.figure.transFigure)
plt.xticks(range(0,42,5))
plt.ylabel('')
plt.xlabel('count')
plt.title('methods')
fig.subplots_adjust(left=0.2)



features.plot.bar(x= 'id', y= 'count', legend= False, ax = ax, title='Features', color = 'green', width = 0.8)
x_legend = '\n'.join(f'{n} - {name}' for n,name in zip(range(len(unique)), unique))
t = ax.text(.7,.1,x_legend, transform=ax.figure.transFigure)
fig.subplots_adjust(right=.60)
plt.ylim(0,9)
plt.xlim(0,46)
ax.set_aspect(5)


plt.show()
