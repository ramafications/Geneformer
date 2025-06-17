
# Running Attention weights on python script bc wont run
import torch
from transformers import BertForMaskedLM
from datasets import load_from_disk
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from geneformer import TranscriptomeTokenizer
import pickle
import os
import anndata as ad
import scanpy as sc
import loompy
import pandas as pd
from collections import defaultdict


# Finding mappings from tokenizer gene to token id so for calculating attention weights

# Initialize the tokenizer
tokenizer = TranscriptomeTokenizer({'stimulation': 'stimulation'}, model_input_size=2048)


token_dict = tokenizer.gene_token_dict
print(f"Token dictionary contains {len(token_dict)} entries")

print("\nSample of token dictionary (first 10 entries):")
sample_items = list(token_dict.items())
for gene, token_id in sample_items:
    print(f"Gene: {gene}, Token ID: {token_id}")

# Create reverse mapping (token ID to gene)
token_to_gene = {}
for gene, token_id in token_dict.items():
    token_to_gene[token_id] = gene

# Print sample of reverse mapping
print("\nSample of token_to_gene mapping (first 10 entries):")
sample_items = list(token_to_gene.items())
for token_id, gene in sample_items:
    print(f"Token ID: {token_id}, Gene: {gene}")

ds = loompy.connect("/u/scratch/r/ramadas/geneformer_tokenized_data/labeled_t_cell_data.loom")

# Extract the stimulation states for all cells
stimulation_states = ds.ca.stimulation
print(f"Extracted {len(stimulation_states)} cell classifications")
print(f"Sample values: {stimulation_states[:10]}")

# Convert to binary labels (assuming 'act' = 1, 'rest' = 0)
cell_labels = np.array([1 if state == 'act' else 0 for state in stimulation_states])
print(f"Sample Values: {cell_labels[:10]}")

import os
import torch
from transformers import BertForMaskedLM
from datasets import load_from_disk
import pickle
import gc  # For garbage collection

model_directory = "/u/scratch/r/ramadas/geneformer_output/k5folds/250505142950/250505_geneformer_cellClassifier_cm_classifier_test/ksplit1/"
input_data_file = "/u/scratch/r/ramadas/geneformer_tokenized_data/Q_data_.dataset"
output_directory = "/u/scratch/r/ramadas/geneformer_models_res/batched_outputs/"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Load model
model = BertForMaskedLM.from_pretrained(
    model_directory,
    output_hidden_states=True,
    output_attentions=True
)
model.eval()
model.to("cuda")

# Load dataset
dataset = load_from_disk(input_data_file)
total_cells = len(dataset)

# Batching parameters
BATCH_SIZE = 100  # Process 100 cells at a time
layer_to_extract = 5

# Process in batches
for batch_start in range(0, total_cells, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_cells)
    print(f"Processing batch {batch_start} to {batch_end-1}")
    
    batch_attention_data = []
    
    # Process cells in this batch
    for i in range(batch_start, batch_end):
        example_cell = dataset.select([i])
        example_cell.set_format(type="torch")
        input_data = example_cell["input_ids"]
        cell_label = "Proliferating" if cell_labels[i] == 1 else "Quiescent"
        
        # Evaluate model
        with torch.no_grad():
            outputs = model(input_ids=input_data.to("cuda"))
            
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # Extract attention weights
                attn = outputs.attentions[layer_to_extract]
                # Map token IDs to gene names
                token_ids = input_data[0].cpu().numpy()
                gene_names = [token_to_gene.get(int(token_id), f"Unknown_{token_id}") for token_id in token_ids]
                
                batch_attention_data.append({
                    'cell_idx': i,
                    'label': cell_labels[i],
                    'label_name': cell_label,
                    'token_ids': token_ids,
                    'gene_names': gene_names,
                    'attention': attn.cpu().numpy()
                })
                
                print(f"Processed cell {i}, Label: {cell_label}")
        
        # Optional: Clear CUDA cache every few cells
        if (i - batch_start + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save this batch
    batch_output_path = os.path.join(output_directory, f"attention_batch_{batch_start}_{batch_end-1}.pkl")
    print(f"Saving batch {batch_start} to {batch_end-1}...")
    with open(batch_output_path, 'wb') as f:
        pickle.dump(batch_attention_data, f, protocol=4)
    
    # Clear memory
    del batch_attention_data
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Saved batch to {batch_output_path}")

print("All batches processed and saved!")


def merge_batches(directory):
    all_data = []
    batch_files = sorted([f for f in os.listdir(directory) if f.startswith("attention_batch_")])
    
    for batch_file in batch_files:
        file_path = os.path.join(directory, batch_file)
        with open(file_path, 'rb') as f:
            batch_data = pickle.load(f)
            all_data.extend(batch_data)
    
    merged_output_path = os.path.join(directory, "merged_attention_data.pkl")
    with open(merged_output_path, 'wb') as f:
        pickle.dump(all_data, f, protocol=4)
    
    print(f"Merged all batches to {merged_output_path}")


merge_batches(output_directory)