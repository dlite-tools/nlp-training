# NLP-Training
## Problem Definition
Training a production model is not trivial, the inference during training and production should have the closest 
behaviour possible such as the pre and post-processing transformations. These transformations should be solid and fully
tested to guarantee that it behaves as expected. During training, the experiments should also be
reproducible and easily trackable.

## Proposal
Our proposal consists of splitting the training project into two main modules, training and inference, allowing to
generate a package for the inference that can be installed as a dependency on the application that serves the model.

### Inference Module
The inference module should contain every piece of code that is strictly necessary to perform an inference, this includes:
- Architectures
- Processor (commonly also known as compose) that given an input applies pre-processing and post-processing transformations.
- Pre-Processing Transformations
- Pos-Processing Transformations
- Utility functions (e.g. load checkpoint into an initiated architecture)

By isolating this module is also easier to create a suite of tests that cover what goes into production. This suite 
should also include testing sequential data transformations, similar to production, to ensure that all pieces work 
together seamlessly.

### Training Module
The training module contains everything else:
- Data augmentation transformations
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
On an ML project life cycle, the transformations applied to data keep suffer changes, so it is important to keep the 
track of all the intermediate steps. This is where Pydantic Model comes in. It is a dataclass model that can be used 
to store the intermediate steps of the pipeline, with automatic type checking and validation, making the pipeline 
transformation easier to follow, maintain and debug.

e.g. In text classification, we can have several parameters to store the raw input information, data transformations 
before the inference, the inference and post-processing transformations.
´´´python
class TextSample(BaseModel):
    raw_text: str
    pre_processed_text: List[str]
    models_input: np.ndarray
    inference_result: np.ndarray
    post_processed_text: str
´´´

#### Constrains
For training a model we need a batch of data, this data must be represented as an array. Since our data currently is stored
in a Pydantic object, and each object represents only one sample, we need an extra step to aggregate the data into a 
unique array. For this we need to get, according to the previous example, the ``models_input`` and stack them.

### How to create a Package
[Poetry](https://python-poetry.org/) is a tool for dependency management and packaging in Python. 
This enables us to generate a package that could be installed via Github easily, so there is no need for a private PyPi 
server. This ensures that the only installed dependencies are strictly necessary to run inference.
