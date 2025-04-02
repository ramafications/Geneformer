# Quiescence Research Project

This repo contains files I used during my research. More detail about these files can be found in the paper

## Please check/update the paths to files used/referenced before running any of these notebooks!

## Cleaning Data:
Use these files in this order to clean and pre process data:
- Celloracle:
    - [concat_mat.ipynb](concat_mat.ipynb)
    - [convert_mat.ipynb](convert_mat.ipynb)
    - [cleaning.ipynb (parts 1 and 2)](cleaning.ipynb)
    - [cleanABC.ipynb](cleanABC.ipynb)
- Geneformer:
    - [concat_mat.ipynb](concat_mat.ipynb)
    - [convert_mat.ipynb](convert_mat.ipynb)
    - [cleaning.ipynb (part 1)](cleaning.ipynb)
    - [geneformer_cleaning.ipynb](geneformer_cleaning.ipynb)

## Needs improvement
### Celloracle foundation model
Currently we are trying to use the ABC model as the foundation model for this, but we originally tried to use a much larger dataset. This would produce better results, but is much harder to process. Here is a link to the celloracle notebook that should be able to process it. Current issue is that the biggest integer you can have in R is less than the number of examples in the dataset we are using so R can't store it. Using this dataset would require some kind of map reduce processing.

### Celloracle ABC Model
The [cleanABC.ipynb](cleanABC.ipynb) file cleans the data into the format [02_atac_peaks_to_TFinfo_with_celloracle_20200801.ipynb](02_atac_peaks_to_TFinfo_with_celloracle_20200801.ipynb) wants. However, when running the motif scan in [02_atac_peaks_to_TFinfo_with_celloracle_20200801.ipynb](02_atac_peaks_to_TFinfo_with_celloracle_20200801.ipynb), I run into issues. It may just be taking a while but I am not sure.

### Geneformer analysis
Despite getting a high classification accuracy from the fine tuned geneformer model, we can't really learn anything from it. Here are some methods we can continue to explore:
- Gradient based analysis
    - Notes in [grads.py](grads.py)
- Tokenization analysis:
    - Code in [tokenization_parse.ipynb](tokenization_parse.ipynb)
    - Finds whether a gene on average belongs to a quiesent or proliferating cell.
    - Uses the tokenized geneformer data
        - This step ranks the genes so we are technically only counting genes for a given cell that geneformer thinks is important
    - Cons:
        - Tokenized geneformer data doesn't depend on the fine tuned model at all, so don't actually use what the model learned.
    - Pros:
        - Very simple
