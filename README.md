---
Sequence generation using Generative Adversarial Networks
---
# Sequence generation using Generative Adversarial Networks


Abstract
---
The GAN framework has recently shown great capabilities in modeling attribute dynamics and we aims to expand the GAN framework to generate highly diverse sequences in an unsupervised manner by only using prior observations. This type of framework can greatly aid testing other systems trough complex simulations or even used for infference by using present data to generate <b>future data </b>.

Main starting point: [ 2019 ] Time-series Generative Adversarial Networks.

Key Challanges
---
Generative adversarial systems traditionally require fine tuning between the modeling power of the generative and adevrsarial part such that one does not overpower the other (similar to sum zero game where each part fights for succes, here we are searching for an equilibrium point in which the generative part generates real looking samples, very hard to clasify from real one by the discriminative part). Chaining multiple time-steps into this setting adds new challanges as both components might use the time dynamics to fool the other component. In a context as this, complex neural architectures that support multi module networks and inter-network communication are a must (as shown in the paper listed above). Some of the key technical components of this project are: 

- Constructing expressive data-loaders for sequence datasets. Time dependence can be expressed both deterministicaly or probabilistical and we, in oder to model time dynamics, give the model acces to both representations.
- Devising a multi-step, multi-module architecture that handles environment representations, time interactions, sample uncertainity and sequence coherence.
- Divising both training regimes and training objectivs that can represent time dynamics and do not simply collapse into <b> statistical ones </b>.
- Developing evaluation methods through which the asses criteria such as: how complex is the generated sequence, how much information variation the generated sequence has, etc.


Progress
---
# 1. Intro to GANs

## Description
The first part of the project was an intro to GANs in which I've created a basic architecture with two multi players perceptrons (MLP) for generative / adversarial elements.

## Implementation
[Simple gan github project](https://github.com/popescuaaa/gan-playground/tree/master/simple_gan)

## Results
Non relevant for this part. Based on visual obervations on generated data and real ones.

## Bibliography
[Generative Adversarial Networks Ian Godfellow et al. 2014](https://arxiv.org/abs/1406.2661)


# 2. Conditional GANs

## Description
The second part of the project was an intro to Conditional GANs. The main idea: starting from the 'simple gan project' listed above, conditionate on additional knowledge the generator and discriminator.
Feed both d and g with: $y$.


## Implementation
[Conditional gan github project](https://github.com/popescuaaa/gan-playground/tree/master/simple_gan)

Is important to mention what techniques for training gans were used from this particular step to speed up the convergence:
- TTsUR: two time scale update rule (different learning rates)
- Alternate training 
## Results
Non relevant for this part. Based on visual obervations on generated data and real ones.

## Bibliography
[Conditional GANs]()


# 3. Recurrent GANs (RGAN) - 2018
# 4. Recurrent Conditional GANs (RCGANS) - 2018
## Description
[ Very close related to final version ]
Starting from the 2017 paper: Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs, I made changes to G and D to make them RNN based (LSTM). I've created a couple of new custom datasets to test the system. Still in development.

The system is conditionate on the lags between time series values at different monents in time.

### Long term goal: 
- [ ] generate future data *
- [ ] create a test enviroment and a set of custom dataloaders to test the TimeGAN and experiment with different techniques *
- [ ] Use next TimeGAN on optical flow on images *

### Short term goals:
- [ ] Correct learning process - currently stuck with constant loss values for G and D
- [ ] Visualize noise influence by plotting MAE / $\lambda$ (change G($\lambda$ * noise, lags))
- [ ] Visualize PCA overlap for generated distribution and real distribution
- [ ] Change loss functions ~ Relativistic GANs
- [ ] Use MNIST as a time series: 28 * 28 => 784 (seq. length)
- [ ] visual validation
    - [ ] plot mean from $n$ generated samples
- [ ] Try to force convergence with gradient penalty (optimal transport + WGAN)


## Implementation
[Conditional gan github project](https://github.com/popescuaaa/gan-playground) - time gan branch

Both G and D are LSTM based.

Each dataset is based on real stock market data and are general purpose (can be used with any csv file that follows the stock market format).

```
# Compute ∆t (deltas)
self.dt = np.array([(self.df[config][i + 1] - self.df[config][i])
                    for i in range(self.df[config].size - 1)])
self.dt = np.concatenate([np.array([0]), self.dt])

# Append ∆t (deltas)
self.df.insert(DATASET_CONFIG.index(config) + 1, 'dt', self.dt)

# Create two structures for data and ∆t
self.stock_data = [torch.from_numpy(np.array(self.df[config][i:i + self.seq_len]))
                   for i in range(0, self.df.shape[0] - self.seq_len)]

self.dt_data = [torch.from_numpy(np.array(self.df.dt[i: i + self.seq_len]))
                for i in range(0, self.df.shape[0] - self.seq_len)]
```

### Current problems
- The learning process is stuck on evolving
    - Solution: the generated sequence is real enough to fool the discriminator and both stop learning. The way to solve this is to change the loss for G and D
- I don't have a specific metric to evaluate the result:
    - Solution: experiment
- Generated values are $0$-centered:
    - Solution: add sequence mean as a conditional parameter for D

## Results
[Wandb Report Draft](https://wandb.ai/popescuaaa/time-gan-2017/reports/RCGAN--Vmlldzo0Mzg2NjY?accessToken=mm3m8h904zgq4xr50xto6rdrpmo8fl2owjv3kq9jhjyikmvid28xo0rdzmq26rzo)

## Bibliography
[Conditional GANs]()


# 5. TimeGAN - 2019




Project Bibliography
---

