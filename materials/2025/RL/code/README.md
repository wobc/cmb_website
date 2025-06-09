# Reinforcement_learning_workshop

To download: https://github.com/FPupillo/RL_workshop_2nd_ed/archive/refs/heads/master.zip

This is an introduction to reinforcement learning of choice data. The material of this repository
has been used for a workshop at the University of Granada on the 9th of June 2025. 
The workshop is divided into two parts: basic concepts and model fitting. 
The slides `reinforcement_learning_slides.pdf"` constitutes a guide through the basic principles and applications of cognitive modeling in general and reinforcement learning models in particular. The script `0.required_packages.R` can be run to install the packages which will be used in the workshop. The .md files `1.basic_concepts.md`, `2.model_fitting.md`,  contain practical examples of the principles illustrated in the slides and can be viewed on github. The corresponding `.Rmd` files can be also run in Rstudio on a computer.
The file `3.exercise_DU_model.Rmd` contains code that can be modified as part of an exercise in which students will create their own model with dual learning rate. 

## Main contents
- Introduction to cognitive modelling
- Introduction to reinforcement learning
- Pavlovian models
- Instrumental models
- Model simulation
- Parameter recovery
- Model recovery
- Model fitting and parameter estimation
- Model comparison
- Model Validation

## Structure of the repository
- `Data` - data used to fit the models 
- `helper_functions` - custom R functions used by the main scripts
- `likelihood_functions` - functions used to compute the likelihood of the choice data, given the models and the parameters
- `model_recovery` - custom scripts to do model recovery
- `output_files` - output of the scripts
- `simulation_functions` - functions used to simulate choice data given the models and the parameters
