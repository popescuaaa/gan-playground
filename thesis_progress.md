---
title: Thesis progress
---

--- 
## Goals:
- [x] Conditional GAN testing
    - [x] improve the model for better performance
    - [x] add visual samples
- [ ] RNN G/D conditional timeGAN
    - [X] custom dataset for stock data
        - [X] basic testing: the system works, but it can be tested
            visually as it's not based on a condition
    - [X] plotly
    - [X] custom dataloader - not needed
        - [ ] complex testing
    - [] testing on condition
---
## main time gan
- [ ]
---
## Achievements:
1. Conditional GAN testing
    - @brief:
    - @result:
        - result for 0.5 dropout on D [ digit 0 - 100 epochs ]
             ![Gan result dummy](./conditional_gan/result_images/dummy_result.png)
        - result for 0.7 dropout on D and 10 training steps for G [ digit 0 - 50 epochs]
            Takes too long and G does not improve over time.
        - result for 0.5 dropout, BCELoss, Discriminator fl Sigmoid and min for G loss [ digit 0 - 50 epochs]
            ![Gan result dummy](./conditional_gan/result_images/bceloss_g_min_condition.png)
    - @docs:
        - [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
2. RNN G/D conditional timeGAN
    - @brief: Starting from [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://paperswithcode.com/paper/real-valued-medical-time-series-generation)
      The main focus was to build a lstm generator and lstm discriminator. The main problem was the dataset which 
      is based on stock market prices for an asset, then the data is processed by extracting the 
      close value for all entries and normalize them (if needed). The second large problem was testing the system
      to be sure that generated data fit in the described time series.
      
    - @testing:
      
        Starting from a series of length 15:
    
       x = <x1, x2, ... , x15> => compute deltas as the two by two differences
       to condition the generator for specific jumps in value
      
      deltas = <d21, d32, ..., d65>
    
    - @result:
    
| Result        |  |   |
| ------------- |:-------------:| -----:|
|    |  |  |
|   |  |  |
|  |  |  |
    


3. Time gan 2
- @brief: 
- @result:
- @docs:
---
Problems:
1. Adapting dataset to current configuration

---
