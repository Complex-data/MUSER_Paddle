# MUSER_Paddle
In this repository, we utilize <a href="https://github.com/PaddlePaddle/Paddle">paddel paddle</a> to implement the MUSER.

## How to set up
* Python3.9 
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```

## Prepare wiki corpus
1) Download the <a href="https://dumps.wikimedia.org/enwiki/latest/">English Wikipedia </a> or <a href="https://dumps.wikimedia.org/zhwiki/latest/">Chinese Wikipedia</a>.
2) Extract the Wikipedia through <a href="https://github.com/attardi/wikiextractor">wikiextractor</a>.
3) Use faiss to index doc and search evidence for the claims, stentence-transformer base model can be found <a href="https://huggingface.co/sentence-transformers">here </a>.

   
## Index a list of documents:
```
python multi_step_retriever_paddle.py --index wiki.jsonl
```


## Test the performance of the trained models
* To test the performance of a trained model, run the command below:
```shell script
python test_trained_model.py --bert_type bert-large
```

The model weights have been converted to paddlepaddle format, you can directly

1) Download the model weights and extract them into the `output/nli_model` folder:

 - <a href="https://drive.google.com/drive/folders/1_JGLjGuVh2ZJhtrmn1yIMtqaJOzka6i1?usp=sharing">NLI model</a>

If you want to train the nli model, run the command below:

```shell script
python train.py --bert_type bert-large
```

For Liar dataset, we provide files that contain evidence retrieved from corpus for training and testing purposes.
