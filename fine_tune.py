import datetime
from geneformer import Classifier
import os

# ================================================================================================================
#
# Original file: https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/cell_classification.ipynb
#
# Notes:
# Editted the file to use the right data source
# I put the notebook into a script because I was having trouble requesting the right GPU and using a notebook at the same time
# After fine tuning, we received an accuracy of 96 percent on the test split (80% train, 20% test)
#
# ================================================================================================================

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

# Current iter with correct data: 96 percent accuracy 
# 240328191631
# 240328
# cm_classifier_test
# /u/scratch/s/schhina/geneformer_output/240328191631

output_prefix = "cm_classifier_test"
output_dir = f"/u/scratch/r/ramadas/geneformer_output/k5folds/{datestamp}"

print(datestamp)
print(datestamp_min)
print(output_prefix)
print(output_dir)
os.system(f"mkdir {output_dir}")

# filter_data_dict={"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]}
training_args = {
    "num_train_epochs": 0.9,
    "learning_rate": 0.000804,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 1812,
    "weight_decay":0.258828,
    "per_device_train_batch_size": 12,
    "seed": 73,
}
cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "stimulation", "states": "all"},
#                 filter_data=filter_data_dict,
                training_args=training_args,
                max_ncells=None,
                freeze_layers = 2,
                # have to change this to 1 for t80/20 or 5 for kfolds
                num_crossval_splits = 5,
                # eval_size=0.2,
                forward_batch_size=200,
                nproc=16)

# previously balanced splits with prepare_data and validate functions
# argument attr_to_split set to "individual" and attr_to_balance set to ["disease","lvef","age","sex","length"]
train_ids = ["1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678", "1561", "1304", "1610", "1430", "1472", "1707", "1726", "1504", "1425", "1617", "1631", "1735", "1582", "1722", "1622", "1630", "1290", "1479", "1371", "1549", "1515"]
eval_ids = ["1422", "1510", "1539", "1606", "1702"]
test_ids = ["1437", "1516", "1602", "1685", "1718"]

train_test_id_split_dict = {"attr_key": "ensembl_id",
                            "train": train_ids+eval_ids,
                            "test": test_ids}

# Example input_data_file: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
# can use Q_data_.dataset for t80/20 or Q_data_k5.dataset for k5
input_data_file = '/u/home/r/ramadas/geneformer_tokenized_data/Q_data_.dataset'
cc.prepare_data(input_data_file=input_data_file,
                output_directory=output_dir,
                output_prefix=output_prefix,
                test_size=0.2)
#                 split_id_dict=train_test_id_split_dict)


train_valid_id_split_dict = {"attr_key": "individual",
                            "train": train_ids,
                            "eval": eval_ids}

# 6 layer Geneformer: https://huggingface.co/ctheodoris/Geneformer/blob/main/model.safetensors
all_metrics = cc.validate(model_directory="/u/home/r/ramadas/scratch_backup/Geneformer",
                          prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
                          id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
                          pval_threshold = 0.05,
                          save_eval_output=True,
                          output_directory=output_dir,
                          output_prefix=output_prefix)
                        #   n_hyperopt_trials=20)
#                           split_id_dict=train_valid_id_split_dict)


cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "stimulation", "states": "all"},
                forward_batch_size=200,
                nproc=16)

all_metrics_test = cc.evaluate_saved_model(
        model_directory=f"{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/",
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
        output_directory=output_dir,
        output_prefix=output_prefix,
    )

# cc.plot_conf_mat(
#         conf_mat_dict={"Geneformer": all_metrics_test["conf_matrix"]},
#         output_directory=output_dir,
#         output_prefix=output_prefix,
#         custom_class_order=["nf","hcm","dcm"],
# )

# cc.plot_predictions(
#     predictions_file=f"{output_dir}/{output_prefix}_pred_dict.pkl",
#     id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
#     title="disease",
#     output_directory=output_dir,
#     output_prefix=output_prefix,
#     custom_class_order=["nf","hcm","dcm"],
# )

print(all_metrics_test)