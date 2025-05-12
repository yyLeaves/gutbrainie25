# Train the best ensemble model
1. `src/tokenize_data_sep.py` tokenizes the data as the desired format for training
2. We got 3 different models using `src/train.py` by setting different random seed (11, 17, 42)
3. We use `src/inference_ensemble_long.py` to ensemble the 3 models and get inference result

# Data augmentation
1. We scraped 500 gut-brain papers from PubMed using `notebooks/scrap_data.ipynb`
2. We ran the inference for augmentation data using best ensemble model in `src/inference_ensemble_aug.py`
3. We tokenize the data using `src/tokenize_data_sep_aug.py`
4. We got 3 different models using `src/train.py` by setting different random seed (11, 17, 42)
5. We get the ensemble result using `src/inference_ensemble_long.py` 

# Tokenized dataset
`train_tokenized_dataset_seperate`: basic train dataset

`train_tokenized_dataset_seperate_aug`: augmented training dataset

`infer_tokenized_dataset`: inference for test set

`train_tokenized_dataset_no_bronze`: retrain model with good quality data only (run4)
