===================
Methodology Concept
===================

Described below is the concept of the work presented. This is based on `Kovacs 
2021 - Finding invariances in sensory coding through gradient methods.`_

.. _`Kovacs 2021 - Finding invariances in sensory coding through gradient methods.`: https://dspace.cuni.cz/handle/20.500.11956/127319


Three main steps take place in order to generate data for finding invariances 
in sensory coding:

    1. Encoding neural responses to stimuli of natural scences in a model; 
    2. Gradient ascent-based snythesis of one optimal stimulus per neuron on the basis of the encoding model;
    3. Gradient descent-based training of a generator model - mapping the latent space to image representation - which is used to snythesize several stimuli per neuron, surpassing a certain neural activation threshold, e.g. greater or equal to the optimal activation, according to the encoding model.


On this basis invariance analyzes are carried out.


Encoding
--------
In the encoding step two type of encodings are computed. For each encoding 
single neuron correlations are computed, so that neurons may be selected for 
further analyzis. These neurons are characterized to be well encoded according 
to a score metric. 


Two types of encoding are used:
    1. Linear encoding:

        .. math:: 
            \mathbf{F} = (\mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{L} )^{-1} \mathbf{X}^{T} \mathbf{y} \\
            \\
            \mathbf{F}: \textrm{Filter representing linear encoding} \\
            \mathbf{X}: \textrm{Image stimuli} \\
            \lambda: \textrm{Regularization factor} \\
            \mathbf{L}: \textrm{Regularization Matrix - for ridge regularization identity} \\
            \mathbf{y}: \textrm{Neural response}\\

        In this case images are currently normalized to zero mean and unit standard deviation as well as globally scaled to be element of [0, 1]. 

    2. DNN encoding: 

        Encoding model is model presented by `Lurz et al. 2021 - Generalizations in Data-Driven Models of Primary Visual Cortex`_.
        In this case images are currently not preprocessed at all. 
.. _`Lurz et al. 2021 - Generalizations in Data-Driven Models of Primary Visual Cortex`: https://www.biorxiv.org/content/10.1101/2020.10.05.326256v2.abstract


For analyization purposes neurons are selected on a score indicating which 
neurons can be encoded very well. To do so, for both encodings:
    1. single neuron correlations are computed;
    2. score is computed: 

    .. math::
        \mathbf{s} = (1 - \mathbf{r}_{RF}) \mathbf{r}_{DNN} \\
        \\
        \mathbf{s}: \textrm{Score}\\
        \mathbf{r}_{RF}: \textrm{single neuron testset correlations of linear encoding}\\
        \mathbf{r}_{DNN}: \textrm{single neuron testset correlations of DNN}\\

    3. The best XXX neurons are further examined. 

..TODO How many neurons to use.

All succeeding computations are carried out for the selected neurons.


Gradient Ascent-based Optimal Stimulus
--------------------------------------
Optimal stimulus computation is based on `Walker et al. 2019 - Inception Loops 
Discover What Excites Neurons Most Using Deep Predictive Models`_.

.. _`Walker et al. 2019 - Inception Loops Discover What Excites Neurons Most Using Deep Predictive Models`: https://idp.nature.com/authorize/casa?redirect_uri=https://www.nature.com/articles/s41593-019-0517-x&casa_token=C0U1ibrLr90AAAAA:akK77Mg0iHzK7Qhr0Fy5E_SRFRGITo35umm7mlU9Ws9BPS0mzhVXhMnRaErwdBnfJDUiFEqYtNJkWyn5HQ

"Most Exciting Images (MEIs)" (Walker et al. 2019) are computed as follows:
    
    1. Generate Gaussian white noise image.
    2. Iteratively add the gradient of the target neuron’s activity using two strategies for dampening high frequencing noise.
        - Blurr the image after every gradient ascent step using a Gaussian filter with a standard deviation that decreases gradually after each iteration:

            .. math::

                \sigma _{t} = \sigma _{0} + \frac{\sigma _{\Delta} t}{n _{iter}}\\
                \\
                \sigma _{t}: \textrm{Standard deviation of Gaussian kernel at } t\\
                \sigma _{0}: \textrm{Initial }\sigma\\
                \sigma _{\Delta} = \sigma _{n _{iter}} - \sigma _{0}\\
                n _{iter}: \textrm{Number of gradient ascent steps}
                
        - Precondition the gradient before adding it to the image with a low-pass filter in the Fourier domain that preferentially suppresses the higher-frequency content of the gradient:
            
            .. math::

                G(\omega _{x}, \omega _{y}) = \frac{1}{(2\pi)^2}(\omega _{x}^2 + \omega _{y}^2)^{-\alpha}\\
                \\
                G: \textrm{Low-pass filter}\\
                \omega _{x}: \textrm{X-Standard deviation}\\
                \omega _{y}: \textrm{Y-Standard deviation}\\
                \alpha: \textrm{gradient smoothing factor}





Generator-based Optimal Stimuli 
-------------------------------
    1. Train generator:
        - train generator model on: n(0,1)
        - optimize neural activation of selected neurons on +- epsilon + MEI (target)
        - second idea: train for: + epsilon + MEI = alpha 
        - thrid idea: update alpha dynamically
    2. Generate Samples:
        - from: u(-2,2)


Invariance Analyzis
-------------------
    1. Compute and apply Region Of Interest (ROI)
    2. Cluster 'most representative/most different' samples
    3. Analyze important samples
