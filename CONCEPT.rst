===================
Methodology Concept
===================

Described below is the concept of the work presented. This is based on `Kovacs 2021 - Finding invariances in sensory coding through gradient methods.`_
Three main steps take place in order to generate data for finding invariances 
in sensory coding: 
Encoding neural responses to stimuli of natural scences in a model; 
Gradient ascent-based snythesis of one optimal stimulus per neuron on the basis
of the encoding model; 
Gradient descent-based training of a generator model - 
mapping the latent space to image representation - which is used to snythesize 
several stimuli per neuron, surpassing a certain neural activation threshold, 
e.g. >= optimal activation, according to the encoding model.
On this basis invariance-analyzis are carried out.

.. _`Kovacs 2021 - Finding invariances in sensory coding through gradient methods.`: https://dspace.cuni.cz/handle/20.500.11956/127319


Encoding
--------
In the encoding step two type of encodings are computed. For each encoding 
single neuron correlations are computed, so that neurons may be selected for
further analyzis. These neurons are characterized to be well encoded according 
to a score metric. 


Two types of encoding are used:
1. Linear encoding:
2. DNN encoding: 
For analyization purposes neurons are selected on a score indicating which 
neurons can be encoded very well:
All sucseeding computations are carried out for the selected neurons.


Gradient Ascent-based Optimal Stimulus
--------------------------------------
Optimal stimulus computation is based on `Walker et al. 2019 - Inception Loops 
Discover What Excites Neurons Most Using Deep Predictive Models`_.

.. _`Walker et al. 2019 - Inception Loops Discover What Excites Neurons Most Using Deep Predictive Models`: https://idp.nature.com/authorize/casa?redirect_uri=https://www.nature.com/articles/s41593-019-0517-x&casa_token=C0U1ibrLr90AAAAA:akK77Mg0iHzK7Qhr0Fy5E_SRFRGITo35umm7mlU9Ws9BPS0mzhVXhMnRaErwdBnfJDUiFEqYtNJkWyn5HQ

"Most Exciting Images (MEIs)" (Walker et al. 2019) are computed as follows:
1. 


Generator-based Optimal Stimuli 
-------------------------------
    5. Train generator:
        - train generator model on: n(0,1)
        - optimize neural activation of selected neurons on MEI (target)
    6. Generate Samples:
        - from: u(-2,2)


Invariance Analyzis
-------------------
    7. Compute and apply Region Of Interest (ROI)
    8. Cluster 'most representative/most different' samples
    9. Analyze important samples
