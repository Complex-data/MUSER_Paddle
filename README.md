````
# MUSER_Paddle

## How to set up
* Python3.7 
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```

## Index a list of documents:
```
python multi_step_retriever_paddle.py --index example_docs.jsonl
```


## Test the performance of the trained models
* To test the performance of a trained model, run the command below:
```shell script
python test_trained_model.py --bert_type bert-large
```

The model weights have been converted to paddlepaddle format, you can directly

1) Download the model weights and extract them into the `output/nli_model` folder:

 - <a href="https://drive.google.com/drive/folders/1_JGLjGuVh2ZJhtrmn1yIMtqaJOzka6i1?usp=sharing">NLI model</a>



## Main experiment setup parameters

| |PolitiFact| Gossipcop| Weibo|
|-|-|-|-|
| Sequence_length | 512|512 |512 |
| Max_encoder_length | 512|512 |512 |
| Min_decoder_length | 64|64 |64 |
| Max_decoder_length | 128|128 |128 |
| Embedding_dimension | 200| 200| 200|
| k(number of paragraphs retrieved) |30 |30 |30 |
| MSR| 0.3| 0.3| 0.3|
|$lambda$ |0.9 |0.9 |0.9 |
| Retrieve_steps | 2| 3| 3|
| Batch_size |64 |64 |32 |
| Maximum_epochs |10 |10 |10 |
| Vocabulary_size | 30522|30522 | 21128 |
| Learning_rate | 1e-5| 1e-5| 1e-5|
````