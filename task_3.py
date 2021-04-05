import numpy as np
import os
import time
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

def calculate_predictions(matrix, p_id_1, p_id_2, k):
    
    # initialise 
    predictions = np.zeros((len(matrix), 1))
    
    for x in (p_id_1, p_id_2):
        parameters = np.load(f'data/GMM_params_phoneme_0{x}_k_0{k}.npy', allow_pickle=True)
        mu = parameters.item(0)['mu']
        s = parameters.item(0)['s']
        p = parameters.item(0)['p']

        predict_x = np.sum(get_predictions(mu,s,p,matrix), axis=1).reshape(-1,1)
        predictions = np.column_stack((predictions,predict_x))

    for index in range(len(predictions)):
        if predictions[index, 1] > predictions[index, 2]:
            predictions[index, 0] = p_id_1
        else:
            predictions[index, 0] = p_id_2
            
    predictions = predictions[:,0]
    return predictions
    

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:,0] = f1
X_full[:,1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
#k = 3
k = 6

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, 
# and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

X_phonemes_1_2 = np.zeros((np.sum(phoneme_id==2)+np.sum(phoneme_id==1), 2))
X_phonemes_1_2 = np.concatenate((X_full[phoneme_id==1,:],X_full[phoneme_id==2,:]), axis=0)

print(np.sum(phoneme_id==2))

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)

#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

########################################/

predictions = calculate_predictions(X_phonemes_1_2,1,2,k)

ground_truth = np.concatenate([np.ones(int(len(predictions)/2)),np.ones(int(len(predictions)/2))*2]).reshape(-1,1)

correct = 0
for x in range(len(ground_truth)):
    if ground_truth[x] == predictions[x]:
        correct += 1

accuracy = (correct/len(ground_truth))
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, 100*accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()