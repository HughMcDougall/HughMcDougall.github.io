  
  
  
Go Back: [Hugh McDougall Astro Blog](.\..\bloghome.html)	&nbsp;	Return to [Blog Home](.\..\bloghome.html)  
  
---  
  
# Comfortably NumPyro  
  
  
  
<p align="center">  
  
  <img width="370" height="217" src="../../images/thumbs/cnpy.jpg">  
  
</p>  
**Navigation**  
* [Getting Started](.\01_gettingstarted\./page.html) - _An absolute beginners guide to NumPyro_  
* [The Constrained Domain](.\02_constraineddomain\./page.html) - _Explanation of constranied vs unconstrained domains in NumPyro_  
* [Testing Different MCMC Samplers](.\03_mcmcsamplers\./page.html) - _A quick overview of the different samplers in numpyro and their strengths / weaknesses_  
* [Nested Sampling](.\04_nestedsampling\./page.html) - _An exploration of nested sampling in NumPyro_  
* [Stochastic Variational Inference](.\06_SVI\./page.html) - _An intro and guide to SVI in NumPyro_  
    * [SVI Part 1](.\06_SVI\01_part1\./page.html) - _Introduction & Explanation_  
    * [SVI Part 2](.\06_SVI\02_part2\./page.html) - _Examples & Comparison with MCMC_  
* [UtilTest](.\07_utils\./page.html) - _An example of using log likelihood utils in numpyro_  
  
---------  
  
  
  
It is a generally accepted fact amongst most reasonable people that Bayesian analysis is the correct approach to most any non trivial statistical problem. In the modern world, we enjoy with a suite of tools that alleviate the tedium of constructing and running these models: Bayesian analysis lets us do things right, probabalistic programming languages (PPLs) let us do it easily, and the magical power of Just-In-Time (JIT) compiled languages allows us to to it _fast_. The [JAX](https://github.com/Joshuaalbert/jaxns)-based PPL [NumPyro](num.pyro.ai/) brings all three together: a python interface that gives great speed and versatility, and makes ubiquitous tasks like parameter constraint or model comparison cheap in both human-time and machine-time.  
  
For the experienced user, the interaction cost hurdle between having an idea and getting a nice ChainConsumer corner plot has never been shorter. The only issue is that some corners of NumPyro can be opaque to the unfamilar user. In this blog, I provide a handful of short and to-the-point tutorials that walk the new user through their first steps into the world of NumPyro, and guide the "almost new" user through the less-obvious features that might otherwise cost hours of scanning documentation.  
  
If you're already confident, you might also consider Dan Foreman Mackey's [Astronomer's Guide to NumPyro](https://dfm.io/posts/intro-to-numpyro/), dive right into the [extensive examples](https://num.pyro.ai/en/stable/) provided by NumPyro's documentation itself.  
  
  
  
---------  
  
This page by Hugh McDougall, 2024  
  
  
  
For more detailed information, feel free to check my [GitHub repos](https://github.com/HughMcDougall/) or [contact me directly](mailto: hughmcdougallemail@gmail.com).  
  
