import pandas as pd
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed
    
    

y = [150, 80, 140, 90, 51, 255] 

color = [str(item/255.) for item in y]


df = pd.read_csv("results/loss_train_imdb_lstm.csv")

# create valid markers from mpl.markers
markers = ['*','s','o','v','+','<']
#cmap = cm.get_cmap('gray')

#df.plot(x=df.index, y=df.columns,style='k--', label='Accuracy')
y = np.zeros((10, 8))
#x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
#x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
x1 = [1,2,3,4,5,6,7,8,9,10]
y1 = (df.iloc[:,1:]).values
y[:,0] = y1[0,:]
y[:,1] = y1[1,:]
y[:,2] = y1[2,:] 
y[:,3] = y1[3,:] 
y[:,4] = y1[4,:] 
y[:,5] = y1[5,:] 
y[:,6] = y1[6,:] 
y[:,7] = y1[7,:]
y = np.array(y)

x_smooth = x1 #np.linspace(x1.min(), x1.max(), 11)
y_smooth = smooth(y,0) #spline(x1,y,x_smooth)
plt.plot(x_smooth, np.array(y_smooth)[:,0], 'turquoise',x_smooth, np.array(y_smooth)[:,1] ,'brown',x_smooth, np.array(y_smooth)[:,2], 'blue',x_smooth, np.array(y_smooth)[:,3] ,'orange',x_smooth, np.array(y_smooth)[:,4], 'green',x_smooth, np.array(y_smooth)[:,5] ,'red',x_smooth, np.array(y_smooth)[:,6] ,'yellow',x_smooth, np.array(y_smooth)[:,7] ,'black')
#plt.plot(x_smooth, np.array(y_smooth)[:,0], 'yellow',x_smooth, np.array(y_smooth)[:,1] ,'black',x_smooth, np.array(y_smooth)[:,2], 'green',x_smooth, np.array(y_smooth)[:,3] ,'red')


# for adding legend
labs = ['Adagrad','AAdagrad','SGD','ASGD','AMSGrad','AAMSGrad','Adam','AAdam' ]

plt.legend(labels=labs,  loc='lower left')
plt.xlabel('Epochs (averaged on 10 trainings)')
plt.ylabel('Train loss ')
plt.title('IMDB movie reviews LSTM')
plt.show()


