
# Dynamical Systems in Machine Learning

Machine Learning, and Deep Neural Networks in particular have produced increasingly impressive successes at solving a wide variety of problems previously thought to be too difficult for computers to solve. Lately, generative models in multiple modalities, text and image in particular have made headlines and are almost certainly only a small preview of things to come.

Before reaching those successes, however, there was a long path of innovations that, in hindsight feel glaringly obvious, but sometimes took years to be found. Many of these can be examined under the lens of dynamical systems, and the purpose of this talk is to present a sequence of vignettes featuring the most salient ones. Note that we explicitly choose to focus on the abstract mathematics view of machine learning, and sweep many meddlesome practical details under the rug. Moreover, we choose to cover a lot of basic concepts and then to point the interesting research that they lead to, rather than going in too much depth.

## What are Neural Networks, again?

A neural network in its simplest form is a parametric function of the form
$$ \Phi(x) = (\varphi_L \circ \dots \circ \varphi_1 )(x), $$
where each function $\varphi_\ell$ is called a *layer*, and is usually of the form
$$ \varphi_l(x_\ell) = \sigma_\ell. \left( W_\ell x_\ell + b_\ell \right), $$
with the weight matrix $W_\ell \in \R^{n_{\ell+1} \times n_\ell}$ and *bias* $b_\ell \in \R^{n_{\ell+1}}$, and the *activation function* $\sigma_\ell$ which is applied pointwise and is typically nonlinear. The matrices $W_\ell$ and biases $b_\ell$ form the *parameters* $\theta$ of the network. The number of layers $L$ is called the *depth*, while the *width* $n_\ell$ of each layer is simply the dimension of its input.

The success of neural networks is owed in great part to their ability to approximate any function given enough layers, as the input gets progressively transformed into high level features that allow the output to be computed. Layer width is known to play a similar but more subtle role. Some of the largest neural networks found today have upwards of thousands of layers, and trillions of parameters.

In applications, neural networks are used to approximate "functions" given as sets of *training* input/output pairs $\{ (x_i,y_i)\}_{i\in I}$. For instance, the inputs $x_i$ may be pictures of various types of animals, and $y_i$ may be labels describing what type of animal picture $x_i$ represents. The goal is to find values of the parameters that induce an approximation the mapping from input to output in a way that generalizes beyond the training dataset. Since there is usually no way to find such parameters by hand, one typically expresses the *training* of the network as an optimization problem
$$ \min_\theta \mathcal{L}(y,x,\theta), $$
with $\mathcal{L}$ the *loss function* which is typically evaluated over the dataset as
$$ \mathcal{L}(y,x,\theta) = \sum_{i\in I} L(y_i, \varphi_\theta(x_i)) $$

The loss function is generally chosen to be differentiable, so that the network may be trained by using optimization algorithms like Gradient Descent, a process which is facilitated by a method known as *Backpropagation*, which simply allows for efficiently evaluating the gradient of the loss function across model parameters by using the chain rule.

## Vignette 1: Optimization Algorithms

This brings us to our first dynamical system. Gradient Descent is quite obviously a dynamical system, and taking the limit as the step size goes to zero, we obtain a gradient flow ODE.

However, plain gradient descent is seldom what gets used in practice as we run into several problems

- As the number of training samples increases to several orders of magnitudes, evaluating the loss function on the whole dataset quickly becomes too expensive. Memory also quickly becomes a concern.
- The loss function is often non-convex, with many symmetries induced by the network's structure, and hence gradient descent can easily get stuck around local minima or saddle points
- The step size is usually not computed using a line search strategy, but rather is taken as a constant that may decrease over iterations, and if chosen incorrectly, the training may converge too early or become unstable due to making too large updates.
- For large models, a single training run may take several days, to several months on a cluster for the largest models.

These problems have spurred an entire [cottage industry](https://www.ruder.io/optimizing-gradient-descent/) of research on faster first order alternatives to gradient descent, as the sheer number of parameters makes higher-order methods like Newton's method prohibitively expensive. A first simple variant being simply to approximate the full loss function by its average over a random subset of training data. This yields the variant called *Stochastic Gradient Descent* (SGD), which may be interpreted as a Stochastic Process in the infinitesimal step size limit, or as an ODE of the average gradient flow. SGD's stochastic nature is widely thought to help escape local minima.

Another interesting modification which may be applied is the addition of *momentum* to the optimization step. Roughly, if a basic gradient descent iteration is given by
$$ \theta_{k+1} = \theta_k - \eta \nabla_{\theta_k} \mathcal{L}, $$
then the version with momentum consists in adding scaled past updates as well, which gives an update of the form
$$ v_{k+1} = \gamma v_k + \eta \nabla_{\theta_k} \mathcal{L} $$
$$ \theta_{k+1} = \theta_k - v_{k+1}. $$
Intuitively, we can think of optimization with momentum as a ball rolling on a surface with acceleration given by the gradient of the surface

This gives rise to a second order differential equation as the step size goes to zero. Indeed, a fascinating line of work showed that momentum methods can be interpreted as discretizations of the equations of motion of a Hamiltonian system. As a result, some of these methods may fail to converge, and instead keep oscillating around a local minimum.

## Vignette 2: Generative Adversarial Networks

Generative Adversarial Networks (GANs) were a very popular approach for tasks like image generation a couple of years ago. Their architecture consists of a pair of networks $\Phi_\theta$ and $\Psi_\phi$, called the *generator* and *discriminator*. The generator is trained to generate random samples from a learned empirical distribution (e.g. a dataset of images of dogs), while the discriminator is trained to detect whether a given sample came from the true distribution or from the generator. The training of these two networks is framed a game, with the generator trying to produce samples that the discriminator can't tell apart from real ones, and the discriminator having the opposite goal. This translates to the following optimization method, called *simultaneous gradient descent*.
$$ \theta_{k+1} = \theta_k - \eta \nabla_{\theta_k} f(\theta_k,\phi_k) $$
$$ \phi_{k+1} = \phi_k - \eta \nabla_{\phi_k} g(\theta_k,\phi_k) $$
Despite their success, GANs are notoriously hard to train for multiple reasons. One of which is simply that as a dynamical system, simultaneous gradient descent is not a gradient flow, as the underlying vector field is not conservative, hence the training process is prone to cycling around optima. 

## Vignette 3: Recurrent Neural Networks
Let us now step back from the dynamical systems involved during training networks, and instead look at the dynamical systems happening *inside* the networks. One the earlier examples of such processes are [*Recurrent Neural Networks*](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks) (RNNs). There are multiple flavors of this approach, but the simplest one takes the form of a discrete dynamical system
$$ x_{n+1} = \Phi(x_n) $$
RNNs were initially used for sequence modelling (language models fall in that category), and a common problem in training them was that the gradients computed during training are prone to converge to zero too quickly or diverge. From the dynamical systems point of view, this is explained by the spectral radius of the Jacobian of $\Phi$ being greater or less than one. As dynamical systems people, we would expect that the optimal conditions would be for spectral radius to be close to one, and that this regime is not far from [Chaos](https://typeset.io/papers/real-time-computation-at-the-edge-of-chaos-in-recurrent-40s7bnqv8q).

## Vignette 4: Residual Networks and Neural ODEs

The previous problem is known as the *vanishing/exploding gradient* problem, and was a major problem with deep neural networks in general. It is adressed in multiple ways in practice, most notably by choosing good activation functions. In this next vignette, however we will focus an approach that spawned an entire new subfield of research, namely *Residual Networks* (ResNets). The key idea of these is what's called *skip connections* and consists in adding a layers input to its output, so that
$$ x_{\ell+1}  = x_\ell + h  \sigma.(W_\ell x_\ell + b_\ell). $$
The intuition behind this is that by adding the layer's input to its output forces it to stay close to the identity map, thereby avoiding stability problems.

The most interesting insight to be had about skip connections, however, is that if you squint just a little bit, it looks *suspiciously like* the Euler discretization of an ODE. This provides another explanation of why they help with the vanishing/exploding gradient problem, as it effectively forces the mapping from input to output to be more regular. From there, we can use other ODE discretization methods to get more regular maps, or directly the "step size" $h$ to be infinitesimally small to obtain a *Neural ODE* of the form
$$ \dot{x} = \Phi(x,t,\theta), $$
where $\Phi$ is a neural network that defines the vector field of the ODE, which can be solved using off-the-shelf numerical ODE solvers. Neural ODEs have been especially useful in settings like physical modeling, time series modeling and [control](https://arxiv.org/abs/2104.05278). Another application consists in [augmenting ODEs](https://arxiv.org/abs/2001.04385) from existing scientific models with neural networks to fit the [missing or unknown terms](https://phys.org/news/2021-11-machine-derive-black-hole-motion.html).

## Vignette 5: Diffusion Models

For our last vignette, we will look at another example of machine learning [taking inspiration](https://www.quantamagazine.org/the-physics-principle-that-inspired-modern-ai-art-20230105/) from dynamical systems to obtain great results, namely Diffusion Models. This is the technique behind image generators like [DALL-E 2](https://openai.com/dall-e-2/) and [Stable Diffusion](https://stability.ai/blog/stable-diffusion-v2-release). To do that, we'll need a bit more explanations.

Diffusion Models (also known as *Denoising Diffusion Models*, or *Score-based generative models*) solve a problem known as *generative modeling*. The idea is that we have some dataset that we assume came from some probability distribution, and we'd like to be able to synthetize new samples from that distribution. GANs are another example of generative models, and were actually the popular choice for image generation before being displaced by Diffusion.

The problem, in the case of images at least, is that the distribution has a complicated, low dimensional support. Images can be thought of as very high dimensional vectors, but the set of images we'd like to generate is a vanishing fraction of all possible images, most of which are pure noise. A common hypothesis is that the "space of pictures" is a low-dimensional manifold in the full space of images.

The key idea of diffusion models is that when starting from a picture and repeatedly adding gaussian noise to it, in the limit we obtain an image that follows a gaussian distribution. At the distribution level, we are starting from our empirical distribution (which we want to approximate), and gradually "blurring it" into a distribution that we know. Mathematically this translates into a *Diffusion Process*, hence the name.

So far, so good, but how do we recover the original distribution? By simply [simulating the reverse process](https://kidger.site/thoughts/score-based-diffusions-explained-in-just-one-paragraph/). Mathematically, the forward process is a very simple Stochastic Differential Equation
$$ dy(t) = dW_t, $$
with initial condition $y(0) \sim \nu$ ($\nu$ being the empirical distribution). Reversing this SDE produces an ODE
$$ dy(t) = -\frac{1}{2} \nabla_y \log p(y(t))dt, $$
which is solved backwards from $y(T) \sim \pi_T$, where $\pi_T$ is the equilibrium distribution of the forward process (usually gaussian noise). The term $\nabla_y \log p(y(t))$ is known in statistics as the *score* of the distribution $p$, and this whole equation can be interpreted once again as a gradient flow. All that is left is to train a neural network $\Phi_\theta$ to approximate the score.

## Closing Thoughts

For a long while, I had a negative impression of Machine Learning as a collection of empirical hacks with little to no well-understood theory. This is still the case in many instances, but what I find remarkable is that we now have people from many fields of mathematics and physics, from Algebraic Geometry to Random Matrix Theory bringing their ideas to this field, so that it has a melting pot of Statistics, Probability, Geometry and Dynamical Systems. In short, it can no longer be said that there no beautiful mathematics in Machine Learning. On the contrary, it is bringing together many fields of mathematics to find new applications.
