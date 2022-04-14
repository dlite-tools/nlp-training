# NLP-Training
## Problem Definition
Training a production model is not trivial, the inference during training and production should have the closest 
behaviour possible such as the pre and post-processing transformations. These transformations should be solid and fully
tested to guarantee that it behaves as expected. During training, the experiments should also be
reproducible and easily trackable.

## Proposal
Our proposal consists of splitting the training project into two main modules, training and inference, allowing to
generate a package for the inference that can be installed as an API dependency that serves the model.

### Inference Module
Inference Module should contain every piece of code that is strictly necessary to perform an inference, this includes:
- Architectures
- Processor (commonly also known as compose) that given an input applies pre-processing and post-processing transformations.
- Pre-Processing Transformations
- Pos-Processing Transformations
- Utility functions (e.g. load checkpoint into an initiated architecture)

By isolating this module is also easier to create a suite of tests that cover what goes into production. This suite 
should also include testing sequential data transformations, similar to production, to ensure that all pieces work 
together seamlessly.

### Training Module
Training Module contains everything else:
- Data augmentation Transformations
- Metrics
- Loss functions
- Dataset, Dataloaders and similar pieces of code
- Code used to train the model, in our case `Trainer`.

This module should also have tests to ensure that the pieces can work together, confirming that the training module is 
fully-functional and behaves as expected.

#### Experimentation
During research is normal to try new approaches, add new transformations, change architectures, etc. So, generally, we 
first do a quick prototype using the available modules to check if it is feasible and if lead to something. These 
experiments are made in a test environment. Then, it might make sense to explore further, so we add the new 
transformation/architecture to the repository, including a suite of tests for the new piece of code, this guarantees 
reproducibility and is already production-ready if it goes well.

### Using Pydantic Model
#### Advantages
Sometimes data transformation previously and pos-inference are pretty complicated. Using Pydantic Model can make our 
problem easier by ensuring that some tasks and saved for later, guaranteeing that a variable is of type x etc.

For example, in this project, we use a Pydantic Model `Document`, which contains information around what has been changed.

#### Constrains
During training will be needed to add some extra step to stack the samples and labels since each sample is linked to a 
model.

### How to create a Package
Poetry is a tool for dependency management and packaging in Python. 
This enables us to generate a package that could be installed via Github easily, so there is no need for a private PyPi 
server. This ensures that the only installed dependencies are strictly necessary to run inference.
