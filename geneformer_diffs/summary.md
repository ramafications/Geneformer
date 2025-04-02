# Edits made to geneformer source code to extract needed inputs for geneformer model

Note: The version of geneformer I used to make these changes may be outdated, so be mindful of that when applying these changes

## emb_extractor.py
- Added the following code to the get_embs function inside the first for loop:
```python
scratch_dir = "" # INSERT YOUR SCRATCH DIRECTORY HERE
torch.save(input_data_minibatch, scratch_dir + "/geneformer_extractions/input_ids.pt")
torch.save(pu.gen_attention_mask(minibatch), scratch_dir + "/geneformer_extractions/atten_mask.pt") 
torch.save(minibatch['stimulation'], scratch_dir + "/geneformer_extractions/labels.pt")
torch.save(original_lens, scratch_dir + "/geneformer_extractions/lens.pt")
print("DONE EXTRACTING INPUTS")
print("SAFE TO EXIT")
```

- Added the following code to the extract_embs method right after "downsampled_data" gets defined:
```python
scratch_dir = "" # INSERT YOUR SCRATCH DIRECTORY HERE
torch.save(downsampled_data, scratch_dir + "/geneformer_extractions/downsampled_data.pt")
```

Note:
Make sure to replace scratch_dir with your scratch directory

## How to apply changes
### Find geneformer install
Find where you installed geneformer. Because I used a conda enviornment, mine was in
```
~/.conda/envs/{ENV_NAME}/lib/{PYTHON_VERSION}/site-packages/geneformer/emb_extractor.py
```

### Apply changes
The above code changes could be hard to follow by themselves, so I included the final file as well.
These are the line numbers where I made my changes:
- 84
- 564

## Use changes
### 