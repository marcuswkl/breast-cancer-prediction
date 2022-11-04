import numpy as np
from matplotlib import cm
def vis2d(ax, model, X_train, Y_train, X_test=[], Y_test=[]):
  # identify graph range
  x_range = [X_train[:,0].min()-0.5, X_train[:,0].max()+0.5]
  y_range = [X_train[:,1].min()-0.5, X_train[:,1].max()+0.5]
  if len(X_test) > 0:
    x_range = [min(x_range[0], X_test[:,0].min()-0.5), max(x_range[1], X_test[:,0].max()+0.5)]
    y_range = [min(y_range[0], X_test[:,1].min()-0.5), max(y_range[1], X_test[:,1].max()+0.5)]
  # create a meshgrid
  xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], .01), np.arange(y_range[0], y_range[1], .01))
  # identify the area of decision
  Z = model.predict([[x,y] for x,y in zip(xx.ravel(), yy.ravel())])
  Z = Z.reshape(xx.shape)
  # plot the decision areas
  ax.contourf(xx,yy,Z,alpha=.8)
  # plot the training and testing data
  ax.scatter([x[0] for x in X_train], [x[1] for x in X_train], c=Y_train, edgecolors='black')
  if len(X_test) > 0:
    ax.scatter([x[0] for x in X_test], [x[1] for x in X_test], c=Y_test, edgecolors='brown', alpha=.8)

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