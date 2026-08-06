# Are Sparse Autoencoders Useful? A Case Study in Sparse Probing
<img width="1213" alt="Screenshot 2025-02-24 at 9 58 54 PM" src="https://github.com/user-attachments/assets/09a20f0b-9f45-4382-b6c2-e70bba6c17db" />

This repository contains code to replicate experiments from our paper [*Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*](https://arxiv.org/pdf/2502.16681). The workflow of our code involves three primary stages. Each part should be mostly executable independently from artifacts we make available:

1. **Generating Model and SAE Activations:**
   - Model activations for probing datasets are generated in `generate_model_activations.py`
   - SAE activations are generated in `generate_sae_activations.py`. Because of CUDA memory leakage, we rerun the script for every SAE, we do this in `save_sae_acts_and_train_probes.sh`, which should work if you just run it.
   - OOD regime activations are specifically generated in `plot_ood.ipynb`.
   - Mutli-token activations are specifically generated in `generate_model_and_sae_multi_token_acts.py`. Caution: this will take up a lot of memory (~1TB).

2. **Training Probes:**
   - Baseline probes are trained using `run_baselines.py`. This script also includes additional functions for OOD experiments related to probe pruning and latent interpretability (see Sections 4.1 and 4.2 of the paper).
   - SAE probes are trained using `train_sae_probes.py`. Sklearn regression is most efficient when run in a single thread, and then many of those threads can be run in parallel. We do this in `save_sae_acts_and_train_probes.sh`.
   - Multi token SAE probes and baseline probes are trained using `run_multi_token_acts.py`.
   - Combining all results into csvs after they are done is done with `combine_results.py`.

3. **Visualizing Results:**
   - Standard condition plots: `plot_normal.ipynb`
   - Data scarcity, class imbalance, and corrupted data regimes: `plot_combined.ipynb`
   - OOD plots: `plot_ood.ipynb`
   - Llama-3.1-8B results replication: `plot_llama.ipynb`
   - GLUE CoLA and AIMade investigations (Sections 4.3.1 and 4.3.2): `dataset_investigations/`
   - AI vs. human final token plots: `ai_vs_humanmade_plot.py`
   - SAE architectural improvements (Section 6): `sae_improvement.ipynb`
   - Multi token: `plot_multi_token.py`
   - K vs. AUC plot broken down by dataset (in appendix): `k_vs_auc_plot.py` 
   
Note that these should all be runnable as is from the results data in the repo.

### Datasets
- **Raw Text Datasets:** Accessible via [Dropbox link](https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/AIocXXICIwHsz-HsXSekC3Y?rlkey=tq7td61h1fufm01cbdu2oqsb5&st=aorlnph5&dl=0). Note that datasets 161-163 are modified from their source. An error in our formatting reframes them as differentiating between news headlines and code samples. 
- **Incorrect class names:** some `news_class` and `disease_class` datasets carry the wrong class name; the labels themselves are unaffected. See the `Notes` column of `data/probing_datasets_MASTER.csv`.
- **Model Activations:** Also stored on Dropbox (Note: Files are large).

## Requirements
We recommend you create a new python venv named probing and install required packages with pip:
```
python -m venv probing
source probing/bin/activate
pip install transformer_lens sae_lens transformers datasets torch xgboost sae_bench scikit-learn natsort
```
Let us know if anything does not work with this environment!


For any questions or clarifications, please open an issue or reach out to us!
