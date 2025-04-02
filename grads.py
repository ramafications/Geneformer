"""
This is my first iteration of using the gradients of the geneformer model to find which genes have the highest
effect on the final output. This approach might work, but I feel like I made some oversights.

To use this file:
- Run the first two cells in examples_extract_and_plot_cell_embeddings.ipynb
    - Make sure to make the changes mentioned in ./geneformer_diffs first
    - If you make these exact changes, ignore any errors. As long as you see 'DONE EXTRACTING INPUTS', it worked
- Get a A100 gpu on hoffman2. 
- If not done already, create a conda enviornment with geneformer, pytorch and numpy. 
- Activate the conda env and run the file

Outputs:
- grad_genes.txt:
    - This is a file containing a dictionary that maps each gene to a score which can be used for further analysis
    - Refer to clean_grads.ipynb to see this file used

Summary of this approach: 
- The model is fine tuned already
- To get access to the inputs the actual model uses, I made some edits to the geneformer files. 
  I included the changes in ./geneformer_diffs
- Find the gradient of the label with respect to a vector gene embeddings (dL/de) by doing a forward and backward
  pass on the model using some data
    - Forward pass needs inputs mentioned in previous step
    - Should probably be test/unseen data, but can be changed
- Aggregrate all the embeddings for a specific gene
    - Currently I do this by just summing up the norm of each individual embedding for each gene
- Save all aggregrated values.
- For each gene, if its aggregrated value meets some threshold, it is probably significant
    - Unsure how to determine whether a specific gene has a higher likliehood of causing quiesence or proliferation from
      just this number. Could use the data to find more info about this.
    - This step is found in clean_grads.ipynb

My thoughts:
- dL/de tells you how to tweak the embedding weights based on how off your prediction was. 
  I was hoping that I could use this gradient to say, "if gene E is expressed, how does that effect the label?", but I 
  don't think I am quite there yet.
- I think something cool could be found using this approach, but more work needs to be done.
  - Ideas for changing input data:
    - Augment the data such that you have identical examples with and without the same gene and see if the norm/label changes
      drastically. If so, that gene might be signficant
    - Only use examples where the model correctly or incorrectly predicts
    - The accuracy of this approach depends on the number of samples for each gene and since we could have a sparse matrix,
      having sufficent input data is neccessary.
"""

import torch
from datasets import load_from_disk
import geneformer
from geneformer import TranscriptomeTokenizer
from geneformer import perturber_utils as pu
from geneformer import emb_extractor as ee
import numpy as np
from torch import nn
import pickle

from collections import defaultdict
from math import ceil

def one_hot(lis):
    nd_arr = np.array(lis)
    tens = torch.zeros([len(lis), 2], dtype=torch.float64)
    act_inds = nd_arr == 'act'
    rest_inds = nd_arr == 'rest'
    tens[act_inds, 1] = 1
    tens[rest_inds, 0] = 1
    return tens

def expand(arr):
    tens = torch.zeros(arr.shape)
    tens[np.arange(arr.shape[0]), torch.argmax(arr, dim=1)] = 1
    return tens

def get_norms(start, end, model, y, loss_fn, token_gene_dict, input_id_tensor, atten_mask_tensor, all_norms):
    """
    all_norms = defaultdict(list)
    """

    # Do forward pass for samples[start:end]
    output = model(input_ids=input_id_tensor[start:end].to("cuda"), attention_mask=atten_mask_tensor[start:end],)
    output.hidden_states[-1].retain_grad()

    # Do backwards pass based on given loss function and true labels (y)
    loss = loss_fn(output.logits, y[start:end].to("cuda"))
    loss.backward()

    # Get gene embeddings and find the norms of each
    hidden_grad = output.hidden_states[-1].grad
    norms = torch.linalg.norm(hidden_grad, dim=2, ord=1)
    named_norms = [] # per sample
    # all_norms = defaultdict(list)

    # Label all gene embedding norms and aggregrate them into a list per gene name
    for i in range(start, end):
        input_ids = downsampled_data_tensor[i]["input_ids"]
        for id_num, token in enumerate(input_ids):
            name = token_gene_dict[token]
            all_norms[name].append(norms[i - start][id_num])
    
    return all_norms

model_type = "CellClassifier"
num_classes = 2
model_directory = "/u/scratch/s/schhina/temp_geneformer_output/240509125208/240509_geneformer_cellClassifier_cm_classifier_test/ksplit1"
batch_size = 20

model = pu.load_model(model_type, num_classes, model_directory, mode="eval")
# model.train()

input_id_tensor = torch.load("/u/scratch/s/schhina/geneformer_extractions/input_ids.pt")
atten_mask_tensor = torch.load("/u/scratch/s/schhina/geneformer_extractions/atten_mask.pt")
labels_tensor = torch.load("/u/scratch/s/schhina/geneformer_extractions/labels.pt")
lens_tensor = torch.load("/u/scratch/s/schhina/geneformer_extractions/lens.pt")
downsampled_data_tensor = torch.load("/u/scratch/s/schhina/geneformer_extractions/downsampled_data.pt")

model_input_size = pu.get_model_input_size(model)
# token_dictionary_file is found in your geneformer install folder
token_dictionary_file = "/u/home/s/schhina/.conda/envs/geneformer_env/lib/python3.11/site-packages/geneformer/token_dictionary.pkl"
with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)

token_gene_dict = {v: k for k, v in gene_token_dict.items()}
pad_token_id = gene_token_dict.get("<pad>")

y = one_hot(labels_tensor)
loss_fn = nn.CrossEntropyLoss()
all_norms = defaultdict(list)

print(input_id_tensor.shape[0])

for i in range(ceil(input_id_tensor.shape[0]/batch_size)):
    get_norms(i*batch_size, min((i + 1)*batch_size, input_id_tensor.shape[0]), model, y, loss_fn, token_gene_dict, input_id_tensor, atten_mask_tensor, all_norms)

useful_genes = {}

for gene_name, norms_value in all_norms.items():
    avg = sum(norms_value)/len(norms_value)
    if avg != 0.0:
        print(f"{gene_name} -> {avg}")
        useful_genes[gene_name] = avg

# print(len(all_norms.keys()))

with open("/u/scratch/s/schhina/grad_genes.txt", "w") as f:
    f.write(f"{useful_genes}")
