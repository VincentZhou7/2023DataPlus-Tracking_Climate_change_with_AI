# Data-Climate-and-AI

**Introduction: Motivation and Research Goal
**
With climate change producing significant impacts to society (such as wildfires and rising sea levels), it is important to inform climate change mitigation and adaptation strategies with evidence for decision-making, including the location and characteristics of energy infrastructure and the status of natural resources and the environment. However, existing data are not always geographically complete or sufficiently up-to-date. 

Our overarching mission is to enable global, automated assessment of energy infrastructure to develop pathways to sustainably address energy needs and climate impacts. To do this, our goal is to develop a technique that can take in text queries, and identify corresponding climate-relevant objects in remotely-sensed data. This will allow researchers to quickly collect valuable information from remotely-sensed data such as location and characteristics of key infrastructure and the impacts on natural resources.

**Challenges
**
However, there are challenges to achieve these goals. 

First, it is difficult to obtain high resolution remote sensing imagery. And even if we have enough data, it is costly to label all of the data by hand. Therefore, the traditional supervised learning approach, which needs a lot of labeled data for training, is not ideal for our project.

In addition, it is also costly to train a relevant large scale AI model from scratch. To solve these challenges, our strategy is to tailor an existing bidirectional text-to-image model by finetuning it using our own, new dataset.

**Datasets**

Before we can finetune and train our model, it is important that we have high quality datasets that we can use for said training. Therefore, we compiled 11 million text-image pairs to enhance the artificial intelligence model training process. 

Text label categories within these images include climate-related features such as airports, power plants, ports, offshore installations, wind turbines, and more. 

These images were previously collected Sentinel-2 satellite images and demonstrate global coverage of select climatic features. 

In our code scripts, we use GeoPandas and spatial join to map coordinate intersections of Sentinel-2 satellite images, and climate-relevant label points/polygons.

TrainingData contains text-image labels for 10/13 of previously collected GeoJSON files. 3/13 are GRIP Polygons (all of “Road” label) and have not been included in TrainingData. 

**Fine Tuning using CLIP
**
The existing model we have applied is CLIP. This model was published by OpenAI in 2021, enabling bidirectional text-image mapping. However, one big shortcoming of the CLIP model is that it performs poorly on climate-related remote sensing images. 

Therefore, our project focused on fine tuning the CLIP model. Our goal was for the base model to learn domain-specific knowledge by feeding it our data. The advantage CLIP has is that it can extract implicit information from our dataset when provided with the image-text pairs that we created. 

The end result is a fine tuned model that can successfully implement text-image queries against remote sensing data and return accurate matches between text and images. 
