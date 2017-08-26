Redefining cancer treatment using MXNet
============

The method used in this submission consists of three parts:
- Get gene information from the internet(variants.py), this can also be done completely offline by download premade gene databases, but the availability online makes simply downloading the specific gene information needed more convenient, in this step the missense variants are also separated into their respective amino acids and possitions, which along with the gene information we can introduce a rough position of the fault
- Use R's tiddytext to extract features using TF-IDF summarizing the lenghty documents(textify.r)
- Final step is to use convolutional neural networks to create and train a model which will classify the enriched text into 9 classes(text_cnn.py)

## Performance
At the time of writing this approach was able to obtain a 0.54178 ranking #30 

## Data
The data is available at [https://www.kaggle.com/c/msk-redefining-cancer-treatment/](https://www.kaggle.com/c/msk-redefining-cancer-treatment/)

## Possible improvements
There are several ways this approach can be improved to get a much better accuracy
- Analyze the extracted words for repeats, badly formatted or unuseful data(big gain)
- Process the enriched variant information using tiddytext as well for a more homogeneus vocabulary(small gain)
- Include more domain knowledge to get more information for each gene and variants(biggest gain)

## Remarks
This competition is quite weird since the purpose is not very clear, you can actually find the ground truth for the testing variants online making it very easy to get a perfect score, the organizers have made a list of external data you can and can't use which limits the approaches you can take to solve this problem, they have stated that this is a text classification competition and as such I treated it like one, still there are places where domain knowledge and external data are absolutely required such as ID59 and ID62 in the training set, the reference text for both of this variants is exactly the same, as well as the gene, making it impossible to accurately predict without external data, this would be a little clearer for the participant if the meaning of the classes were disclosed, which are likely loss of fucntion and loss of function for this case.

## Run
Due to the way windows and linux process encoding, the results may vary depending on the OS you use.
- Rscript textify.r
- python variants.py
- python text_cnn.py --gpus 0 --num-embed 500 --dropout 0.5
- python eval.py

## References
- [Gene cards gene information](http://www.genecards.org/cgi-bin/carddisp.pl?gene=PTPRT)
- [Some initial protein analysis](https://www.kaggle.com/danofer/genetic-variants-to-protein-features)
- [Mutation information](http://cancer.sanger.ac.uk/cosmic/mutation/overview?id=133823)
- [More gene information](https://www.ncbi.nlm.nih.gov/gene/4780)
- [Very insightful variant information](https://varsome.com/variant/hg19/PTPRT%3AR1209W)
- [TF-IDF analysis](https://www.kaggle.com/headsortails/personalised-medicine-eda-with-tidy-r)
- [MXNet text cnn](https://github.com/apache/incubator-mxnet/blob/master/example/cnn_text_classification/README.md)