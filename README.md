# SMILES generation with a LSTM RNN

## Project Layout

```
.
├── src/
│   ├── preprocess.py   
│   ├── tokens.py       
│   ├── dataset.py      
│   ├── model.py       
│   ├── train.py        
│   ├── generate.py
│   ├── plot_history.py     
│   └── stats.py        
│   └── outputs/
├── data/
│   ├── smiles_clean.txt
│   └── smiles_train.txt
├── outputs/
├── environment.yml     
├── train.slurm         
└── README.md
```

## Approach

The model utilized is a multi-layer long short-term memory (LSTM) RNN. 

The approach of this project borrows the basis of the following papers:

>  ***Bidirectional Molecule Generation with Recurrent Neural Networks***; F Grisoni, M Moret, R Lingwood, and G Schneider (2020); *DOI: 10.1021/acs.jcim.9b00943*

>  ***Generative Recurrent Networks for De Novo Drug Design***; A Gupta, AT Müller, BJH Huisman, JA Fuchs, P Schneider, G Schneider (2018); *DOI: 10.1002/minf.201700111*


> ***Molecular Generation with Recurrent Neural Networks (RNNs)***; J Bjerrum and R Threlfall (2017); *ArXiv abs/1705.04612*


### Theoretical Background

LSTM RNN is a specialized neural network for sequential data. 


LSTMs enable backpropagation of the error through time-steps hence helping preserve them through feedback connections that pass on information of as it propagates forward.

![image info](./assets/gate_of_lstm.webp)

The information which is added or removed to the cell state is regulated by structures called **gates**. These consist of a neural network layer and a pointwise operation.

The **forget gate** is a sigmoid layer that decides which information should be kept or removed from the cell state. It uses the current input and previous hidden state then applies a sigmoid function to generate values between 0 and 1.

The **input gate** is a sigmoid layer that decides which value to update and store in the cell state. A tanh function creates a vector of new candidate values that could be added to the cell state.

The **output gate** is a sigmoid layer that determines which information from the current cell state should be passed as the hidden state (output) at the current time step. 

This **hidden state** is then passed to the next time step and can also be used for generating the output of the network.

### Pre-processing

### LSTM Training



|  Parameter                       | Description                | Reasoning                           | 
|-------------------------|----------------------------|-------------------------------------|
| Architecture            | 2-layer LSTM, hidden = 512 | Literature: commonly 1 - 3 Layers.                   |           
| Embedding               | Learned Embedding          | Learned embeddings potentially let the model discover and encode chemical similarity.  |           
| Vocabulary/tokenization | 99 atom-wise tokens        | Atom-in-SMILES replaces generic SMILES tokens with environment-aware atomic tokens, reducing token degeneration and improving chemical translation accuracy. [Source](https://hunterheidenreich.com/notes/chemistry/molecular-representations/notations/atom-in-smiles-tokenization/)                         |          
| Training Data           | 1,27M provided SMILES      | Default Dataset Received in Course           |           
| Epochs                  | 30                         | Literature: 10 - 50 epochs, picked the middle.                                 |           
| Batch size              | 256                        | Mini-batch gradient descent: A middle ground for available GPU utilization.                                 |           
| Learning rate           | 0.001                      | *Kingma and Ba*, **2014**                    |         
| Dropout rate           | 0.2                    | Literature: commonly 0.1 - 0.5.                 |  



### Molecule Generation

## Evaluation Metrics: Fréchet ChemNet Distance (FCD)

## Extras 1: Metaheuristic Hyperparameter Improvement

## Extras 2: Best-of visualizations and insights

## Resources

https://medium.com/geekculture/10-hyperparameters-to-keep-an-eye-on-for-your-lstm-model-and-other-tips-f0ff5b63fcd4 

https://hiya31.medium.com/a-guide-to-lstm-hyperparameter-tuning-for-optimal-model-training-064f5c7f099d

https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html

https://www.youtube.com/watch?v=6niqTuYFZLQ

https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/ 

https://hunterheidenreich.com/notes/chemistry/molecular-representations/notations/atom-in-smiles-tokenization/ 

## AI-use disclaimer

Claude 
Opus 4.7 on various levels of effort

Uses:
- code proofreading and bug fixes
- theoretical understanding of ML
- suggestions for project structure
- pseudocode generation 