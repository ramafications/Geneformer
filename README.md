Procedure
Run /geneformer_cleaning.ipynb using this https://ucla.app.box.com/file/1506606455985?s=rccifmuc6l7nvrsyt21wqbsaas9hynlw


note: I skipped cleaning and concatenation given I did not have access to the original dataset, so I ran directly on the cleaned and concatenated loom file\
also cells to run cross validation of k=5 making a drop out tokenization file
Run fine_tune.py on new tokenized file to generate both cross validation models and t80/20


note: must change local base geneformer model to adjust for bad code see longer documentation
Run Metric_Results.ipynb to generate statistics and both models as well as permuted data by modifying orginal tokenized datset


Run attention_weights.ipynb on the model we created in step 2 and the dataset in order to extract attention weights on the evaluation on each step


note: this is a sample on small subset in order to extract the attention matrix of every tcell in dataset use attention_matrix_extract.py to get all the attention weights
Run EmbExtractor.ipynb to do 3 things:


Get visual representations of the embeddings themselves on a 2d view
build an svm model that qualifies the embeddings based on their boundary distance
in silico perturbation (stats have not been run can refer to this: https://geneformer.readthedocs.io/en/latest/api.html#in-silico-perturber-stats to build it)
Run weight_embedding_analysis.ipynb


load boundary distances and attention rankings to see if relative similarly classified cells have genes with similar attention scores
note: the current method is to bin them by distances and see how the rankings change within each bin
the longer documentation has some potential ideas but now since the attention weights and boundary distances are extracted there are so many ways where you can search for patterns between them

