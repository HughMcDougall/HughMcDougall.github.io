Previous Entry: [Nested Sampling](.\..\04_nestedsampling\page.html)	&nbsp;	   
  
  
Go Back: [Comfortably NumPyro](.\..\blog_numpyrohome.html)	&nbsp;	Return to [Blog Home](.\..\..\bloghome.html)  
  
# Stochastic Variational Inference  
**Navigation**  
* [SVI Part 1](.\01_part1\./page.html) - _Introduction & Explanation_  
* [SVI Part 2](.\02_part2\./page.html) - _Examples & Comparison with MCMC_  
  
---------  
  
  
  
When it comes to Bayesian Inference, the general purpose tool is almost always MCMC, or some other marginalization / integration engine: some method to explore parameter space and map out likelihood contours with no a-prior knowledge of the _shape_ of these contours. Of course, in many practical cases we _do_ have prior knowledge of the distribution shape, at least vaguely. In a given problem, we might know that one or more of the parameters will be constrained to correlated Gaussians, or independent exponentials, or some other mix of the various "canonical" distributions. In such cases, we can leverage this prior knowledge about the "shape" of the posterior distribution to fit a simplified, "good enough", approximate distribution. This method has broadly been named _Variational Inference_, and the general-case method has come to bear the name of **Stochastic Variational Inference** (SVI).  
  
The general idea of SVI is to have some complicated posterior distribution, $p(z)$, and approximate it with some simple "surrogate distribution", $q(z \vert \phi)$, which is 'similar' to $p(z)$. Here, $\phi$ are tuneable variables of our surrogate distribution (_not_ model parameters) e.g. the mean and width of a normal distribution. A more informative name might be something like "Surrogate Distribution Optimization of Parameters", as our goal is to find the $\theta$ that makes $q_{\theta}(z)$ fit $p(z)$ as closely as possible.   
  
The core benefit of SVI is that we turn the _integration_ problem of grid-search or MCMC into an _optimization_ problem. Optimizations problems scale better with dimension and are easier to run, and so we can use SVI to get (at the cost of precision) good speed up on any number of high dimensional problems. SVI isn't a small topic to cover, so I've split this entry into two parts. In the first, I introduce the reader to the basics: the broad ideas of SVI and the maths that underpin them, as well as some step-by-step examples of what SVI tools are available in NumPyro and how to use them. In the second part, I zoom out to more involved and practical problems, comparing SVI to MCMC in speed and robustness, and testing some of its more advanced features.  
  
  
---------  
  
This page by Hugh McDougall, 2024  
  
  
  
For more detailed information, feel free to check my [GitHub repos](https://github.com/HughMcDougall/) or [contact me directly](mailto: hughmcdougallemail@gmail.com).  
  
