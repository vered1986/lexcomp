## Evaluating Text Representations on Lexical Composition


### Dependencies

- Python 3
- argparse
- [allennlp (0.8.1)](https://github.com/allenai/allennlp/)

### Downloading Data:

Download the pre-trained models using `bash download.sh`.

The VPC classification and LVC classification tasks need a copy of the BNC corpus. 
Please download the XML version from [here](http://www.ota.ox.ac.uk/desc/2554),
and update its path in the JSON files.

Once you do, you will need to extract the sentences themselves:

```
python preprocessing/get_sentences_from_bnc.py \ 
    [/path/to/corpora]/bnc/2554/download/Texts/ \ 
    diagnostic_classifiers/data/vpc_classification/ \ 
    diagnostic_classifiers/data/vpc_classification
```


### Running experiments:

To train all the models for a given task, e.g. NC literality, run:

```
bash diagnostic_classifiers/experiments/nc_literality/train.sh 
```

To evaluate:

```
bash diagnostic_classifiers/experiments/nc_literality/evaluate.sh
``` 

To get the predictions for the test set:

```
bash diagnostic_classifiers/experiments/nc_literality/predict.sh
```

### Adding a new task:

You will need to create a directory under experiments 
with the JSON files specifying the architecture and hyper-parameters.
Each model requires a `DatasetReader`, `Model`, and a `Predictor`. 
You can use the ones implemented in this repository or implement
new ones according to the specific model's needs. 
  
See the [AllenNLP tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/walk_through_allennlp/configuration.md)
for additional instructions on configuring models.

If you'd like to create new data, follow the 
[preprocessing instructions](preprocessing/README.md).  

### Adding a new representation:

You will need to implement a new `TokenIndexer` and `TokenEmbedder` or 
`TextFieldEmbedder`. The first takes a sequence of words and returns
their IDs, and the second gets the IDs and returns the vectors.
Look at the implementations in this repository and in the 
AllenNLP repository, and read the [documentation](https://github.com/allenai/allennlp/blob/master/tutorials/notebooks/data_pipeline.ipynb) 
there. 

You will also need to add a JSON file for the task + representation
combination and add the command to the train/evaluate/predict 
bash files. 

### Citation


[Still a Pain in the Neck: Evaluating Text Representations on Lexical Composition](https://arxiv.org/pdf/1902.10618.pdf)

Vered Shwartz and Ido Dagan. arXiv 2019.


