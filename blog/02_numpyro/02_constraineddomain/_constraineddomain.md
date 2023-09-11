# The Constrained / Unconstrained Domains & Getting Likelihoods from MCMC

The moment you start trying to do anything fancy with NumPyro, it's only a matter of time before you find yourself running into issues with the constrained and unconstrained domain. NumPyro uses JAX's autodiff for a lot of its functionality: these gradients don't make sense if you have any distribution / prior with discontinuities, and so NumPyro internally performs a coordinate transformation from these _discontinuous_ distributions to a new domain with _continuous_ distributions. The danger comes in the fact that _some_ of NumPyro's features want arguments specified in this unconstrained domain, but it's not always easy to tell which. 

In this example, we cover a simple 1D example of how to transform parameters between the constrained and unconstrained domain and how to use this to similarly adjust a likelihood function, and then apply these techniques to an MCMC chain to show how you can use NumPyro's normal MCMC methods to also report the likelihood at each sample.


```python
# Statistical + JAX Tools
import jax, numpyro
import jax.numpy as jnp
from numpyro.infer.util import transform_fn

# Utility, Math & Plotting
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
```

## The Constrained & Unconstrained Domain

In this section, we look at how a simple uniform distributed parameter, $x \sim U(2,2)$, is re-paramaterized by numpyro into a domain where the sharp discontinuities at the edges of this prior are removed. In the following cell, we perform the transformation manually. First, we need to make sure we have the `transform_fn` function. We'll also define the uniform distribution as its own separate python object, as we need to feed it into our transformation later.


```python
from numpyro.infer.util import transform_fn

dist = numpyro.distributions.Uniform(0,2) # The NumPyro distribution that is going to get transformed
def model_test():
    x = numpyro.sample('x', dist)
```

Now the steps for transforming from the "constrained" domain, i.e. the "real" values of x, to the "unconstrained" domain. First, we generate a range of 'x' values that span the values allowed by the uniform distribution, $x \in [0,2]$, then we transform these in a two-step process:
1. First, we need to tell NumPyro what distributions we're working with in the dictionary `transforms`
2. Then we apply these transformations with `transform_fn`, specificying which values we're transforming with the second argument and that we're pulling from the unconstrained to the constrained domain with `invert=True`
This second step transforms the entire dictionary and returns as another dictionary with similar keys, and so we need to extract the specific key `'x'`. Notice that, if we had many variables, we could transform them all in one go by making a dictionary with all parameters / transformations at once.

By plotting the transformation, we can see that it does two things:
1. The transformed values are _unconstrained_, i.e. $x_{uncon} \in (-\infty,\infty)$
2. They are 'normalized' such that the "central" value is $x_{uncon} = 0$


```python
transforms = {"x": numpyro.distributions.biject_to(dist.support)}

x_con = np.linspace(0, 2, 1024)
x_uncon = transform_fn(transforms, {'x': x_con}, invert=True)['x']

#---------------------------------
# Plotting
plt.plot(x_con, x_uncon, c='k', lw=3)
plt.axline( [1,0], slope=2, zorder = -1, c='k', ls=':', alpha = 0.5)
plt.xlabel("Constrained x Value")
plt.ylabel("Unconstrained x value")
plt.grid()
plt.show()
```


    
![png](output_5_1.png)
    


Converting the variables between the constrained and unconstrained domains is easy enough, as shown above, but feeding these directly into a likelihood function won't give us the right answer. Instead, we need to weight by the *derivative* of this transformation to recover the correct probability. If you're note familair with how probability distributions change with coordinate transformations, the key idea is that corresponding differential elements have the same "power" contained in them:

\begin{equation}
    P(x_{con}) \cdot dx_{con} = P(x_{uncon}) \cdot dx_{uncon}
\end{equation}

Such that the distribution transforms like:

\begin{equation}
    P(x_{con}) = P(x_{uncon}) \cdot \frac{dx_{uncon}}{dx_{con}}
\end{equation}

As a first pass, we'll do this using crude finite differences:


```python
diff = (x_uncon[2:]-x_uncon[:-2]) / (x_con[2:]-x_con[:-2])
```

Now we feed our unconstrained parameters into the likelihood function, which we access from the `potential_energy` utility function that NumPyro gives us. This potential energy returns the **negative log likelihood**. For our model `model_test()`, this looks something like:

$PE(x_{uncon}) = -ln(\mathcal{L(x_{uncon})})=$ `numpyro.infer.util.potential_energy(model_test, model_args=(), model_kwargs={}, params={'x': x})`

Even though we have no model `args` or `kwargs`, these fields still have to be explicitly given as empty tuples like above. In a more complicated case with data, e.g. `model_with_data(X,Y,E)`, these would be fed into the `model_args` field.

Using this approach, we can evaluate our likelihoods in the unconstrained domain, and then correct them back to the constrained domain with our derivative:


```python
# Get potential energies
PE = [numpyro.infer.util.potential_energy(model_test, model_args=(), model_kwargs={}, params={'x': x}) for x in x_uncon]
PE = np.array(PE)

# Convert to likelihood
L = np.exp(-PE)

# Weight by transformation derivative
y= L[2:] * diff

# True model for comparison
y_true = 0.5*(x_con[2:]>0.0)*(x_con[2:]<2.0)
y_true[0] = 0
y_true[-1]= 0

# Plot
plt.plot(x_con[2:], y, c='blue', lw=2, label = "$\mathcal{L}$ from finite diff")
plt.plot(x_con[2:], y_true, c='c', lw=1, zorder=-1, label = "$\mathcal{L}$ True")

plt.legend()
plt.grid()
plt.xlabel("$x_{con}$")
plt.ylabel("$\mathcal{L}(x) (Approx)$")
plt.show()
```


    
![png](output_9_0.png)
    


Looking above, we can see that the likelihood function plateaus at $\mathcal{L}\approx 0.5$ for most of the domain, and is constrained to $x \in [0,2]$, both of which are what we expect from our $x \sim U(0,2)$ distribution. The unusual behaviour at the edges of the domain is a result of our poor esimate of $\frac{dx_{uncon}}{dx_{con}}$, coupled with the extreme asymptotic gradients at $c \in \{0,2\}$. 

Fortunately, we can use JAX's native autodifferentiation to get an *analytically accurate* derivative function. First, re-define the transformation function to be a bit easier to read, and then apply jax's auto-diff to this, which is as easy as using `jax.grad(function)`:


```python
def tform(x):
    out = transform_fn(transforms, {'x': x}, invert=True)['x']
    return(out)

tform_diff = jax.grad(tform)
```

Now take these two functions, along with the likelihood evaluation we saw before, and bring it all together in a single function:


```python
def l_uncon(x):
    xdash = tform(x)
    diff = tform_diff(x)
    
    ll  = numpyro.infer.util.potential_energy(model_test, model_args=(), model_kwargs={}, params={'x': xdash} )
    out = jnp.exp(-ll) * diff
    return(out)

l_uncon = jax.vmap(l_uncon)
```

In the last line, the `l_uncon(x)` function has been `vmapped` by JAX, making it an efficienct vectorized function. Just like using JAX's `jax.jit(function)` decorator, we can only transform functions that are at the top level. Now, we can apply this function to an entire sequence of values without issue:


```python
# Evaluate and set any nan (out of domain) values to zero
y = l_uncon(jnp.array(x_con))

# -------------------
y = jnp.nan_to_num(y, copy=False, nan=0.0) # A mask that stops errors at the poorly defined boundaries

# Plot
plt.plot(x_con, y, c='blue', lw=2)
plt.ylim(ymin=0, ymax = np.max(y)*2)
plt.grid()
plt.xlabel("$x_{con}$")
plt.ylabel("$\mathcal{L}(x)$")
plt.show()
```


    
![png](output_15_0.png)
    


In this example we transformed _into_ the unconstrained domain, but transforming back out is as simple as swapping `invert=True` to `False`. Knowing how to convert between the two domains is important for cases like running multiple MCMC chains with different start locations and retrieving sample likelihoods from an MCMC chain. In the following section, we cover an example of this second application.

# Using Domain Transforms to get MCMC Sample Likelihood

In this section we tackle two problems: 
1. Firstly, how to get a NumPyro MCMC sampler to record information about the likelihood of the MCMC samples
2. Secondly, how to use the information about the constrained / unconstrained domain from the previous section to convert this information into a useful value

To keep things simple, we'll use a simple unimodal gaussian and uniform priors. In this example, we set up our NumPyro model slightly differently to normal: instead of using a the `numpyro.distributions.Normal` prior for $x$ and $y$, we'll instead use _uniform_ priors and use `numpyro.factor`, which lets as multiply our posterior distributio  by some arbitrary factor / function. Given our model is gaussian, the two approaches are equivalent (plus or minus a normalization constant) and we're only doing this here so that we have a readily evaluable probability function to test against down the track.


```python
#----------------------
# Parameters / prior volume
xmin, xmax = -5, 15
ymin, ymax = -8, 8
V0 = (xmax-xmin) * (ymax-ymin) # Prior vol
#----------------------

# Probability func w/o prior
def prob_func(x,y):
    out=0
    out+=jnp.exp(-1/2 * ((x)**2+y**2))
    return(out)

# Log Probability function
def log_prob_func(x,y):
    return(jnp.log(prob_func(x,y)))

# Numpyro model + prior distributions
distx = numpyro.distributions.Uniform(xmin,xmax)
disty = numpyro.distributions.Uniform(ymin,ymax)
def np_model():
    x = numpyro.sample('x', distx)
    y = numpyro.sample('y', disty)

    numpyro.factor(name="prob_fac", log_factor = log_prob_func(x,y))
```

### Generating MCMC-Like Chain

Now fire a standard MCMC run at this using NUTS. In a simple distribution like this, a single chain should be fine. Note that, when running the MCMC sampler, we instruct it to also log "`extra_fields`" like the potential energy etc. We can do this for **any value tracked by the sampler state**. e.g. we're using `NUTS`, a type of HMC, so we can instruct NumPyro to track any of the values listed in the [HMCstate](https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.HMCState) class (e.g. `z_grad`: the grad vector or `i`, the itteration number)


```python
# Construct the NUTS sampler
MCMC_sampler = numpyro.infer.MCMC(
    numpyro.infer.NUTS(np_model), 
    num_chains=1,
    num_warmup=1000, 
    num_samples=int(10000))

# Run and acquire results, also storing potential energy
MCMC_sampler.run(jax.random.PRNGKey(1), 
                 extra_fields=("potential_energy",)
                 ,)
MCMC_results = MCMC_sampler.get_samples()
print("Sampling done")
```

In HMC, potential energy is proportional to $\chi^2$:

\begin{equation}
    U_x(x) = -ln|\mathcal{L_x(x)}|=\frac{-\chi^2}{2}
\end{equation}

NumPyro uses this terminology consistently across all of its samplers, e.g. the sampler adapative sampler ([SAstate](https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.sa.SAState)) also refers to 'potential energy' despite not actually being formulated in kinetic terms. As an aside, the 'extra fields' tuple can sometimes play up if you don't have a comma after the last argument. E.g. `extra_fields=("potential_energy",)` will work, but `extra_fields=("potential_energy")` won't. 

Now, plot these results in chainconsumer to confirm everything is working correctly. As expected, we see a bivariate gaussian with $\sigma_x=\sigma_y=1$


```python
c = ChainConsumer()
c.add_chain(MCMC_results, name = "MCMC")
c.plotter.plot(extents = {'x':[xmin,xmax], 'y':[ymin,ymax]})
plt.show()
```


    
![png](output_22_0.png)
    


Any extra info is stored in a dictionary accesible through `sampler.get_extra_fields()`. I'm also going to calculate the *true* likelihood of each point to demonstrate an issue with this approach.

When we plot the two against eachother, we see the proportionality is _almost_ right, but something is clearly going wrong.


```python
# Extract info from MCMC sampler object
extra_fields = MCMC_sampler.get_extra_fields()
pot_en = extra_fields['potential_energy']

# Get true likelihood for all points for comparison / demonstration purposes
X, Y = MCMC_results['x'], MCMC_results['y']
Ltrue = np.array([prob_func(x,y) for x,y in zip(X, Y)]) / V0 # factor of V0 represents attenuation by the prior
logLtrue = np.log(Ltrue)

#--------------------------
# Plotting
plt.scatter(logLtrue, -1 * pot_en, s=1)
plt.axline([0,1], slope=1,c='r',zorder=-1)
plt.grid()
plt.xlabel("True Likelihood")
plt.ylabel("$-2 \\times $Potential Energy")
plt.show()
```


    
![png](output_24_0.png)
    


The issue here is simple: this `potential energy` is defined in terms of the **unconstrained parameter space**, and so we need to do a transformation back to real parameter space using a conversion factor $U_\theta(\theta) = U_x(x) \cdot D$:

\begin{equation}
    D=\Pi_i \left( \frac{dx'_i}{d\theta_i} \right)
\end{equation}

This is simple enough to do, all have to do is:
1. Get the transformation functions from $x\leftrightarrow\theta$
2. Take their gradient using `jax.grad`
3. Get a function that takes their product to get the conversion factor

In the following, I just JAX's `jax.vmap` so that our final function can process all of the MCMC chain at once, this time using applying JAX's function transformations using decorators:


```python
transforms = {"x": numpyro.distributions.biject_to(distx.support), "y": numpyro.distributions.biject_to(disty.support)}

@jax.grad
def _tformx(x):
    out1 = transform_fn(transforms, {'x': x}, invert=True)['x']
    return(out1)
    
@jax.grad
def _tformy(y):
    out2 = transform_fn(transforms, {'y': y}, invert=True)['y']
    return(out2)
    
@jax.vmap
def tform_diff_xy(x,y):
    out = _tformx(x) * _tformy(y)
    return(out)
```

Equipped with this, we can easily get the required scaling for each sample and then apply this factor to transform `potential energy` into a true log_likelihood.  Plotting the two against eachother, we can confirm that this new transformed value is the property that we're after:


```python
# X,Y are coordinates from MCMC chain
diff_scaling = tform_diff_xy(X,Y) # get coordinate change scaling
L_from_MCMC = np.exp(-pot_en) * diff_scaling # Convert potential energy to likelihoods

# plot against direct likelihood
plt.scatter(Ltrue,L_from_MCMC, s=1)
plt.axline([0,0], slope=1, c='r',zorder=-1)
plt.axis('square')
plt.grid()
plt.xlabel("True Likelihood")
plt.ylabel("Likelihood from Potential Energy")
plt.show()
```


    
![png](output_28_0.png)
    

