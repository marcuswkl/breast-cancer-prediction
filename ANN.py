import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import cm

def vis3d(fig, model, X_train, Y_train, X_test=[], Y_test=[]):
  possible_class = np.unique(Y_train)
  y_range = [0, 1]
  y_data_min = X_train.min(axis=0)
  y_data_max = X_train.max(axis=0)
  if len(X_test) > 0:
    y_data_min = np.amin([y_data_min, X_test.min(axis=0)], axis=0)
    y_data_max = np.amax([y_data_max, X_test.max(axis=0)], axis=0)
  single_y = np.arange(y_range[0], y_range[1], .1)
  single_y = single_y.reshape(len(single_y), 1)
  yy = []
  for i in range(X_train.shape[1]):
    if len(yy) == 0:
      yy = np.tile(single_y,1)
    else:
      old = np.tile(yy, (single_y.shape[0],1))
      new = np.repeat(single_y, yy.shape[0])
      new = new.reshape(len(new),1)
      yy = np.hstack([new, old])
  yy_data = [[yi*(y_data_max[i] - y_data_min[i])+y_data_min[i] for i,yi in enumerate(y)] for y in yy]
  zz = model.predict(yy_data)
  train_x = (X_train - y_data_min)/(y_data_max - y_data_min)
  axes = []
  for i in possible_class:
    ax = fig.add_subplot(len(possible_class), 1, i+1)
    ax.plot(yy[zz == i].transpose(), c=cm.Set2.colors[i%cm.Set2.N], alpha=0.5)
    ax.plot(train_x[Y_train == i].transpose(), c='black', lw=5, alpha=.8)
    ax.plot(train_x[Y_train == i].transpose(), c=cm.Dark2.colors[i%cm.Set2.N], lw=3, alpha=.8)
    ax.set_title("output = {}".format(i))
    ax.set_xticks([i for i in range(X_train.shape[1])])
    ax.set_ylim(y_range)
    axes.append(ax)
  return axes
    
def visualise(mlp):
  # get number of neurons in each layer
  n_neurons = [len(layer) for layer in mlp.coefs_]
  n_neurons.append(mlp.n_outputs_)

  # calculate the coordinates of each neuron on the graph
  y_range = [0, max(n_neurons)]
  x_range = [0, len(n_neurons)]
  loc_neurons = [[[l, (n+1)*(y_range[1]/(layer+1))] for n in range(layer)] for l,layer in enumerate(n_neurons)]
  x_neurons = [x for layer in loc_neurons for x,y in layer]
  y_neurons = [y for layer in loc_neurons for x,y in layer]

  # identify the range of weights
  weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

  # prepare the figure
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  # draw the neurons
  ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
  # draw the connections with line width corresponds to the weight of the connection
  for l,layer in enumerate(mlp.coefs_):
    for i,neuron in enumerate(layer):
      for j,w in enumerate(neuron):
        ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'white', linewidth=((w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)*1.2)
        ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'grey', linewidth=(w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)

    
# Main method
if __name__ == '__main__':
    data = pd.read_csv('Breast_cancer_data.csv')
    
    # Slice data into input and targets 
    input_data = data.iloc[:,:5]
    target = data['diagnosis']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_data, target, train_size=0.7)
    
    # Data preprocessing
    # Use standardization to normalize data cuz range of diff features are different
    # Scale input data based on training input data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Construct ANN model
    # Construct feedforward neural network with 1 hidden layer of 7 neuron and max iteration of 2000
    mlp = MLPClassifier(hidden_layer_sizes=(7), activation="relu",max_iter=2000)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    visualise(mlp)
    
    fig = plt.figure()

    #vis3d(fig, mlp, X_train, y_train, X_test, y_test)

    # Evaluate performance of ANN model 
    print(confusion_matrix(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))
    
