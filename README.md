# Custom-Optimizer-on-Keras
ASGD, AAdaGrad, Adam, AMSGrad, AAdam and AAMSGrad - See below for details about this Accelerated-optimizers

### Requirements

* Keras version >= 2.0.7 )
* Python >= 3 
  
  
### How to use
```python
opt = Adam(amsgrad = False)
network.compile(optimizer= opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

```

### Results 
I included 3 files to test the optimizers on mnist data (logistic + MLP), cifar10 (CNN) and IMDB (LSTM).
To run the code, simply go to command line an put ```python mlp.py``` or ```python logistic.py```. 
The results are saved in different csv files. You can use matplotlib or any other library you want to plot the results.

You will notice in the code that i put these 2 lines :
```python 
#network.save_weights('initial_weights_log.h5')
network.load_weights('initial_weights_log.h5')
```
As I wanted to compare the optimizers, I kept the same initialization for the weights. In the first line I save the weigths (just once) and for the rest of the experiments I just read the weigths from the file. 

### Accelerating First-Order optimization algorithms (pdf is available)
Several stochastic optimization algorithms are currently available. In most cases, selecting the best optimizer for a given problem is not an easy task as they all provide adequate results. Therefore, instead of looking for yet another ’absolute’ best optimizer, accelerating existing ones according to the context might prove more effective. This paper presents a simple and intuitive technique to accelerate first-order optimization algorithms. When applied to first-order optimization algorithms, it converges much more quickly and achieves a better minimum for the loss function when compared to traditional algorithms. The proposed solution modifies the update rule, based on the variation of the direction of the gradient and the previous step taken during training. Several tests were conducted with SGD, AdaGrad, Adam and AMSGrad on three public datasets. Results clearly show that the proposed technique, as tested in the field, has the potential to improve the performance of existing optimization algorithms.

 It is implemented as follows (AAdam)
```python 
m_t = tf.where(tf.logical_and((past_g* g) >=0, tf.abs(past_g - g)>S),(self.beta_1 * m) + (1. - self.beta_1) * (g+past_g),(self.beta_1 * m) + (1. - self.beta_1) * g) 
```
```past_g``` is the previous value of the gradient. ```past_g* g``` tells us if the direction of the gradient has changed. Accelerated versions will take bigger steps only if the gradient did not change its direction. Once ```tf.abs(past_g - g)``` reaches a certain value, Accelerated algorithms stops adding ```past_g``` to the update step until the end of the training. We only change ```m_t``` and not ```v_t``` as the goal is to take bigger steps. 

This technique can be applied to any first order optimizers. 
