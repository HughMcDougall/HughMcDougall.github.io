# Statistical Methods: Important but not Scary
## OzGrav ECR Workshop 2024 

This is a rough transcript of a talk I presented to the early career researcher (ECR) workshop at the OzGrav 2024 retreat, hosted at the University of Queensland. You can download the slides directly [here]().


![jpg](./Slide1.JPG)  

This is aimed at graduate students and early career post-docs, aiming to cover two main areas:
1. That the tendency to treat statistical methods as black boxes can lead to dangerous failure modes 
2. Despite their reputation, stats methods are actually quite easy to understand, and even a basic understanding equips you to know what tools will / won't fail in different jobs.

![jpg](./Slide2.JPG)

Part one: why stats methods are important, and something we, as scientists, need a functional understanding of even if they aren't the main focus of our work.

![jpg](./Slide3.JPG)

In physics, we're in the business of working with _models_. When we first start learning in highschool, things are nice and simple: we have some physics that we describe with a mathematical model, and we combine this with data to gain some understanding about reality. Of course very quickly gets more complicated: the universe has randomness, our data has noise, are models are approximate, and so we need _statistical_ models to bridge the gap between our theory and our data. For all but the simplest cases, these statistical models are too complicated to get results by hand, and so we more often than not also need some _numerical_ model to actually do the calculations, and finally use those results to make some conclusions about physical reality. Sow what we is this hierarchy of modelling, each layer stacked on top of one another.

The danger is that it's really only these first and last layers that are Capital P __physics__, and there can be this problem where these middle two blocks go under-examined, with people relying on existing pre-made tools to handle their stats and their calculations without looking too closely. This forms a "blind-spot" in our models, a zone of ambivalence that people don't alot much attention. 

If there's one thing I want to leave you with today, it's that sometimes things will go wrong in this blind-spot, and that will follow through to your results. If you're lucky, your final results will look like garbage and you'll know you have to go back to the drawing board. If you're _un_-lucky, your tools can fail _invisibly_, giving results that look completely reasonable but are actually completely wrong. If you don't know what to look for, if you don't know how these tools work and how they break, you can end up this incorrect results forwards and affecting your physics.

![jpg](./Slide4.JPG)  

To demonstrate that this is a real thing that happens and not some scary story I've made up to frighten you, I'm going to share a cautionary tale from my neck of the woods in Reverberation Mapping. So, what exactly is that? If you're not familiar, Reverberation Mapping is a clever trick where we use time-domain observations to infer the masses of the super-massive black holes at the core of distant galaxies. 

In brief: every galaxy as a a super-massive black hole at its core, and we want to know how heavy they are because this helps us answer all sorts of questions. These galactic nuclei are involved in galaxy evolution through AGN feedback, there's open questions about how they initially formed and how they act in galaxy mergers, all things we can probe by looking at how the masses of the SMBH population has evolved over cosmological time.

The issue, of course, is that SMBH's dark, compact and extremely far away: it's very hard to measure their masses directly. Fortunately, we're in luck. Some of these black holes have matter falling onto them, scrunching down into these enormous, ultra-hot, accretion disks, easily producing more light than the rest of the host galaxy. These _active_ galactic nuclei are bright enough to see at cosmological distances.

![jpg](./Slide5.JPG)  

So how do we use this to get the mass? Well, an AGN has more than just the black hole and disk, it has this entire complex geometry that builds up around it. For our purposes today, the AGN has only two parts: the accretion disk that throws of enormous amounts of light, and this fast-orbiting cloud around it called the broad line region that captures some of that light and re-emits it as emission lines. The width of the lines tells us how fast the orbit is, and so basic high-school level physics lets us get the central mass of the black hole, but only if we know the orbital radius.

That's where revberation mapping comes in. The accretion disk isn't a static light source: it flickers wildly, AGN are highly variable. This flickering is "echoed" off the broad line region in the same pattern, but with a delay between the two because it takes time for the light to travel. If we can observe both objects for long enough, building up light-curves and measuring that delay, we can use this lag as a ruler to measure the scale of the system, and from there get the mass.

![jpg](./Slide6.JPG)

If you entered revberation mapping ten years ago, you would think that this was a solved problem. We have the physics which I just described, and we have a statistical model that describes the AGN light-curves as something called a damped random walk, a particular stochastic process that explains the properties of the flickering really well. This damped random walk is an example of a Gaussian process, which means we can simulate it, which means we can build a full Bayesian generative model and do Bayesian statistics.

There's software that does this: a program called JAVELIN that fits the lag, plus a half dozen other parameters, and does its fitting using Markov Chain Monte Carlo (MCMC). Specifically, it uses the package program `emcee`, which you may have used or heard of as it's probably the most popular MCMC package in astrophysics. 

![jpg](./Slide7.JPG)  

So what's the problem? What's changed in the last ten years? What's changed more than anything is the sorts of surveys we do revberation mapping with. For early surveys you'd track a handful of nearby sources, at low redshift, really intensely for a couple of months. Now we work with in __industrial scale__ surveys like OzDES or the Sloan Digital Sky Survey (SDSS), which track hundreds to thousands of AGN out to deep redshifts over multiple years, working at lower precision and cadence but making up for it in depth and shear weight of numbers.

The problem with these multi-year surveys is that, when you track any one AGN, you get half-yearly seasonal gaps in your observations owing to the Sun being in the way for half of that time. These seasonal gaps give rise to the problem of __aliasing__.

![jpg](./Slide8.JPG)  

![jpg](./Slide9.JPG)  

![jpg](./Slide10.JPG)  

![jpg](./Slide11.JPG)  

![jpg](./Slide12.JPG)  

![jpg](./Slide13.JPG)  

![jpg](./Slide14.JPG)  

![jpg](./Slide15.JPG)  

![jpg](./Slide16.JPG)  

![jpg](./Slide17.JPG)  

![jpg](./Slide18.JPG)  

![jpg](./Slide19.JPG)  

![jpg](./Slide20.JPG)  

![jpg](./Slide21.JPG)  

![jpg](./Slide22.JPG)  

![jpg](./Slide23.JPG)  

![jpg](./Slide24.JPG)  

![jpg](./Slide25.JPG)  

![jpg](./Slide26.JPG)  

![jpg](./Slide27.JPG)  

![jpg](./Slide28.JPG)  

![jpg](./Slide29.JPG)  

![jpg](./Slide30.JPG)  

![jpg](./Slide31.JPG)  

![jpg](./Slide32.JPG)  

![jpg](./Slide33.JPG)  

![jpg](./Slide34.JPG)  

