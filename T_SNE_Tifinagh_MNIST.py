# Visualization of the Tifinag-MNIST database using the T-SNE algorithm

## The libraries we will use
"""

import time
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

"""## Data loading and adaptation"""

def upload_data(path_name, number_of_class, number_of_images): 
    X_Data = []
    Y_Data = []
    for i in range(number_of_class):
        images = os.listdir(path_name + str(i))
        for j in range(number_of_images):
            img = cv2.imread(path_name + str(i)+ '/' + images[j], 0)
            X_Data.append(img)
            Y_Data.append(i)
        print("> the " + str(i) + "-th file is successfully uploaded.", end='\r') 
    return np.array(X_Data), np.array(Y_Data)

n_class = 33
n_train = 2000

x_data, y_data = upload_data('/media/etabook/etadisk1/EducFils/PFE/DATA2/train_data/', n_class, n_train)

x_data = x_data.astype('float32')
x_data = np.reshape(x_data, (x_data.shape[0], 28*28))
x_data /= 255
print('x_data shape:', x_data.shape)
print(x_data.shape[0], 'data samples')

"""## Convert images and label vector to a Pandas DataFrame"""

feat_cols = [ 'pixel'+str(i) for i in range(x_data.shape[1]) ]
df = pd.DataFrame(x_data,columns=feat_cols)
df['y'] = y_data
df['label'] = df['y'].apply(lambda i: str(i))
x_data, y_data = None, None
print('Size of the dataframe: {}'.format(df.shape))
df.head()

"""## Displaying images from the Dataframe"""

np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

plt.gray()
fig = plt.figure( figsize=(18,12) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Letter: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
plt.show()

"""## Launch of the T-SNE algorithm


"""

N = 50000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

"""## Visualisation"""

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 33),
    data=df_subset,
    legend="full",
    alpha=0.3
)