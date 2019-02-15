# Custom-Optimizer-on-Keras
Adam, AMSGrad and AAdam (or AAMSGrad - See below for details about this optimizer) optimizers 

### Requirements

* Keras version >= 2.0.7 )
* Python >= 3 or higher
* Your computer :-)
  
  
### How to use
```python
opt = Adam(amsgrad = False)
network.compile(optimizer= opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

```

### Results 
I included 2 files to test the optimizers on mnist data (logistic + MLP).
To run the code, simply go to command line an put ```python mlp.py``` or ```python logistic.py```. 
The results are saved in different csv files. You can use matplotlib or any other library you want to plot the results.

You will notice in the code that i put these 2 lines :
```python 
#network.save_weights('initial_weights_log.h5')
network.load_weights('initial_weights_log.h5')
```
As I wanted to compare the optimizers, I kept the same initialization for the weights. In the first line I save the weigths (just once) and for the rest of the experiments I just read the weigths from the file. 

### You want to know about AAdam right ?
AADAM (or AAMSGrad or even ASGD) means  Accelerated Adam. The idea is intutive and straightfoward to implement.
Lets take the example of the ball rolling down a hill. If we consider that our objective is to bring a ball (parameters of our model) to a lowest elevation of a road (cost function), what we do is to push a little bit more that ball in the direction of the current gradient and the past gradient when we know that the past update was in the same direction. This is done by taking the maximum or the sum between the normal update (taken by any optimizer) and the the previous update. We consider that an update is a value which represents a slope of a vector. The ball will gain more speed as it continues to go in the same direction and looses its current speed as soon as it passes over a minimum. Once the direction change, we stop accelerating the ball and let the optimizer takes the full control of the rest. This new update rule accelerates the move of the ball towards the minimum (local or global depending on where we started). It is implemented as follows (AAdam)
```python 
m_t = tf.where(tf.logical_and((tf.sign(p_grad) * tf.sign(g)) >=0, tf.equal(mem_t, 0)),(self.beta_1 * m) + (1. - self.beta_1) * (g+p_grad),(self.beta_1 * m) + (1. - self.beta_1) * g) 
```
```p_grad``` is the previous value of the gradient. ```mem_t``` tells us if the direction of the gradient changed at least one time, if yes its value is 1 else it is 0. This memory is important because, AAdam will take bigger steps only if the gradient did not change its direction since the begining. Once ```mem_t``` changed to 1, AAdam stops adding ```p_grad``` until the end of the training. We only change ```m_t``` and not ```v_t``` as the goal is to take bigger steps. 

This technique can be applied to any optimizers. 
