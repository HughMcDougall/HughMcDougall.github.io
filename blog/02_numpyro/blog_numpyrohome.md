Previous Entry: [WebGuide](.\..\01_webguide\page.html)	&nbsp;	   
  
  
Go Back: [Hugh McDougall Astro Blog](.\..\bloghome.html)	&nbsp;	Return to [blog home](.\..\bloghome.html)  
  
---------------------------------------------------------------------------  
* [NumPyro - Getting Started](.\01_gettingstarted\./page.html)  
* [MCMC Samplers](.\02_mcmcsamplers\./page.html)  
* [WebGuide](.\03_nestedsampling\./page.html)  
* [Constrained Domain](.\04_constraineddomain\./page.html)  
* [WebGuide](.\05_parallelizing\./page.html)  
  
  
# Comfortably NumPyro  
  
![png](../../images/thumbs/cnpy.png)  
  
It's a generally accepted fact amongst most reasonable people (see: people who agree with me) that Bayesian analysis is the correct approach to most any non trivial statistical problem. In the modern world, we're equipped with a suite of tools that alleviate the tedium of constructing and running these models: Bayesian analysis lets us do things right, probabalistic programming language interfaces let us do it easily, and the magical power of JIT compiled languages allows us to to it _fast_. The [JAX](https://github.com/Joshuaalbert/jaxns)-based PPL [NumPyro](num.pyro.ai/) brings all three together: a python interface that gives great speed and versatility that makes problems like MCMC and model comparison cheap in both human-time and machine-time.  
  
In an age of fast personal computers and probabalistic programming languages, the interaction cost gap between having an idea and getting a nice ChainConsumer corner plot has never been shorter. There is only one hurdle: NumPyro is, to the unfamiliar user, constructed with a intuitive and human-friendly design and documentation rivaled only by the zodiac cypher. In this blog, I provide a handful of short and to-the-point examples that walk the new user through their first steps into the world of NumPyro, and guide the "almost new" user through the less-obvious features that otherwise might cost hours of fruitlessly trawling stack exchange.  
  
If you're already confident, you might also consider Dan Foreman Mackey's [Astronomer's Guide to NumPyro](https://dfm.io/posts/intro-to-numpyro/), dive right into the [extensive exampeles](https://num.pyro.ai/en/stable/) provides by NumPyro's documentation itself.  
---------  
  
This page by Hugh McDougall, 2023  
  
  
  
For more detailed information, feel free to check my [GitHub repos](https://github.com/HughMcDougall/) or [contact me directly](hughmcdougallemail@gmail.com).  
  
