from joblib import Parallel, delayed
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeavePOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import random
import xgboost
from utils_data import get_xy_traintest
from sklearn.model_selection import RandomizedSearchCV
import sklearn
import time

try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False

# TRAINING UTILS


def get_cv(X_train):
    n_samples = X_train.shape[0]
    if n_samples <= 12:
        cv = LeavePOut(2)
    elif n_samples < 128:
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    else:
        val_size = min(int(0.2 * n_samples), 100)  # 20% of data or max 100 samples for validation
        train_size = n_samples - val_size
        cv = [(list(range(train_size)), list(range(train_size, n_samples)))]
    return cv

def get_splits(cv, X_train, y_train):
    # Generate only splits where validation has at least one of each class
    if hasattr(cv, 'split'):
        splits = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            if len(np.unique(y_train[val_idx])) == 2:  # Ensures both classes in the validation set
                splits.append((train_idx, val_idx))
    else:
        splits = cv  # For predefined list-based splits
    
    return splits


def find_best_reg(X_train, y_train, X_test, y_test, plot=False, n_jobs=-1, parallel=False, penalty="l2", seed = 1, return_classifier = False):
    # Determine cross-validation strategy
    best_C = None
    if X_train.shape[0]>3: # cannot reliably to cross val. just going with default parameters
        cv = get_cv(X_train)
    
        Cs = np.logspace(5, -5, 10)
        avg_scores = []

        def evaluate_fold(C, train_index, val_index):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
            
            if penalty == "l1":
                model = LogisticRegression(C=C, penalty="l1", solver="saga",random_state=seed, max_iter=1000)
            else:
                model = LogisticRegression(C=C, random_state=seed, max_iter=1000)
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            return roc_auc_score(y_fold_val, y_pred_proba)

        for C in Cs:
            splits = get_splits(cv, X_train, y_train)
            # Parallelize the inner loop using joblib
            if parallel:
                fold_scores = Parallel(n_jobs=n_jobs)(delayed(evaluate_fold)(C, train_index, val_index) for train_index, val_index in splits)
            else:
                fold_scores = [evaluate_fold(C, train_index, val_index) for train_index, val_index in splits]

            avg_scores.append(np.mean(fold_scores))

        # Find the index of the best score (max for classification, min for regression)
        best_C_index = np.argmax(avg_scores)
        best_C = Cs[best_C_index]

    # Train final model with best C
    metrics = {}

    if best_C is not None:
        if penalty == "l1":
            final_model = LogisticRegression(C=best_C, penalty="l1", solver="saga",random_state=seed, max_iter=1000)
        else:
            final_model = LogisticRegression(C=best_C, random_state=seed, max_iter=1000)
    else:
        if penalty == "l1":
            final_model = LogisticRegression(penalty="l1", solver="saga", random_state=seed, max_iter=1000)
        else:
            final_model =  LogisticRegression(random_state=seed, max_iter=1000)
    # Shuffle X_train and y_train based on seed
    rng = np.random.RandomState(seed)
    shuffle_idx = rng.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
    if best_C is not None:
        metrics["val_auc"] = np.max(avg_scores)
    else:
        metrics["val_auc"] = roc_auc_score(y_train, final_model.predict_proba(X_train)[:, 1])
    if plot:
        plt.semilogx(Cs, avg_scores)
        plt.xlabel("Inverse of Regularization Strength (C)")
        met1_name, met2_name = 'auc', 'auc'
        plt.ylabel(f'{met1_name} on validation data')
        plt.title(f'{"Logistic Regression"} Performance vs Regularization\nBest C = {best_C:.5f}; {met2_name} = {metrics[met2_name]:.2f}')
        plt.show()
    if return_classifier:
        return metrics, final_model
    return metrics



def find_best_pcareg(X_train, y_train, X_test, y_test,plot=False, max_pca_comps=100):
    # Standardize the data
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Determine the range of PCA dimensions to try
    max_components = min(X_train.shape[0], X_train.shape[1], max_pca_comps)
    pca_dimensions = np.unique(np.logspace(0, np.log2(max_components), num=10, base=2, dtype=int))

    # Fit PCA for the maximum number of components
    pca = PCA(n_components=max_components)
    X_combined_pca_full = pca.fit_transform(X_combined_scaled)

    best_score = -float('inf')
    best_model = None
    best_n_components = None
    metrics = {}
    if X_combined_pca_full.shape[0] > 3:
        cv = get_cv(X_train)
        scores = []

        for n_components in pca_dimensions:
            X_pca = X_combined_pca_full[:, :n_components]
            fold_scores = []
            splits = get_splits(cv, X_train, y_train)
            for train_index, val_index in splits:
                X_fold_train, X_fold_val = X_pca[train_index], X_pca[val_index]
                y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

                model = LogisticRegression(random_state=42, max_iter=1000)

                model.fit(X_fold_train, y_fold_train)
            
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                fold_scores.append(roc_auc_score(y_fold_val, y_pred_proba))

            avg_score = np.mean(fold_scores)
            scores.append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_model = LogisticRegression(random_state=42, max_iter=1000).fit(X_pca, y_train)
                best_n_components = n_components
                metrics['val_auc'] = best_score
    else:
        best_n_components = X_combined_pca_full.shape[0]
        best_model = LogisticRegression(random_state=42, max_iter=1000).fit(X_combined_pca_full, y_train)
        y_train_pred_proba = best_model.predict_proba(X_combined_pca_full)[:, 1]
        metrics['val_auc'] = roc_auc_score(y_train, y_train_pred_proba)

    # Transform test data using PCA
    X_test_pca = pca.transform(X_test_scaled)[:, :best_n_components]

    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_pca)

    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = best_model.predict_proba(X_test_pca)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
    
    
    if plot and X_combined_pca_full.shape[0]>3:
        plt.semilogx(pca_dimensions, scores)
        plt.xlabel("Number of PCA Components")
        plt.xscale('log', base=2)
        met1_name, met2_name = 'auc', 'auc'
        plt.ylabel(f'{met1_name} on validation data')
        plt.title(f'Best PCA dimension: {best_n_components}, {met2_name} = {metrics[met2_name]:.2f}')
        plt.show()

    return metrics

def find_best_knn(X_train, y_train, X_test, y_test, plot=False, n_jobs=-1):
    # Standardize the data
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
    metrics = {}
    if X_train.shape[0] > 3:
        # Determine the range of k values to try
        cv = get_cv(X_train)
        test_split = get_splits(cv, X_train, y_train)
        min_train_vals = float('inf')
        for split in test_split:
            min_train_vals = min(len(split[0]), min_train_vals)
        max_k = min(X_train.shape[0] - 1, 100, min_train_vals)  # Limit max k
        k_values = np.unique(np.logspace(0, np.log2(max_k), num=10, base=2, dtype=int))

        best_score = -float('inf')
        best_model = None
        best_k = None
        scores = []

        # Function to evaluate a fold
        def evaluate_fold(k, train_index, val_index):
            X_fold_train, X_fold_val = X_combined_scaled[train_index], X_combined_scaled[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            return roc_auc_score(y_fold_val, y_pred_proba)
        # Loop over k values
        for k in k_values:
            splits = get_splits(cv, X_train, y_train)
            # Parallelize the cross-validation folds using joblib
            fold_scores = Parallel(n_jobs=n_jobs)(delayed(evaluate_fold)(k, train_index, val_index) 
                                                for train_index, val_index in splits)
            avg_score = np.mean(fold_scores)
            scores.append(avg_score)

            # Update best score and model
            if avg_score > best_score:
                best_score = avg_score
                best_model = KNeighborsClassifier(n_neighbors=k).fit(X_combined_scaled, y_train)
                metrics['val_auc'] = best_score
                best_k = k
    else:
        best_k = 1
        best_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_combined_scaled, y_train)
        y_train_pred_proba = best_model.predict_proba(X_combined_scaled)[:, 1]
        metrics['val_auc'] = roc_auc_score(y_train, y_train_pred_proba)
    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_scaled)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)

    if plot and X_train.shape[0] > 3:
        plt.semilogx(k_values, scores)
        plt.xlabel("Number of Neighbors (k)")
        plt.xscale('log', base=2)
        met1_name, met2_name = 'auc', 'auc'
        plt.ylabel(f'{met1_name} on validation data')
        plt.title(f'Best k: {best_k}, {met2_name} = {metrics[met2_name]:.2f}')
        plt.show()

    return metrics

def find_best_xgboost(X_train, y_train, X_test, y_test, classification=True, binary = True, plot=False, cv_folds  =3):
    # Check if X_train has less than 3 samples

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if len(X_train) <= 3:
        # Return default values without performing random search
        default_model = xgboost.XGBClassifier()
        default_model.fit(X_train_scaled, y_train)
        y_test_pred = default_model.predict(X_test_scaled)
        metrics = {}
        y_train_proba = default_model.predict_proba(X_train_scaled)[:, 1]
        metrics['val_auc'] = roc_auc_score(y_train, y_train_proba)
        metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
        metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
        y_test_pred_proba = default_model.predict_proba(X_test)[:, 1]
        metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
        return metrics

    # Define the hyperparameter space for random search
    param_space = {
        'n_estimators': np.arange(50, 300, step=50),
        'max_depth': np.arange(2, 6),
        'learning_rate': np.logspace(-3, -1, num=10),  # log scale for eta
        'subsample': np.linspace(0.7, 1.0, num=5),
        'colsample_bytree': np.linspace(0.7, 1.0, num=5),
        'reg_alpha': np.logspace(-3, 1, num=10),  # log scale for L1 regularization
        'reg_lambda': np.logspace(-3, 1, num=10),  # log scale for L2 regularization
        'min_child_weight': np.arange(1, 10)
    }

    # Cross-validation
    cv = get_cv(X_train)
    splits = get_splits(cv, X_train_scaled, y_train)

    best_auc = 0
    best_params = None 
    n_iter = 10
    for _ in range(n_iter):
        params = {k: random.choice(v) for k, v in param_space.items()}
        model = xgboost.XGBClassifier(**params, eval_metric='logloss')
        
        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in splits:
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train_scaled[val_idx]
            y_fold_val = y_train[val_idx]
            model.fit(X_fold_train, y_fold_train)
            y_fold_val_proba = model.predict_proba(X_fold_val)[:, 1]
            cv_scores.append(roc_auc_score(y_fold_val, y_fold_val_proba))
        mean_auc = np.mean(cv_scores)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    metrics = {}
    best_model = xgboost.XGBClassifier(**best_params, eval_metric='logloss')
    best_model.fit(X_train_scaled, y_train)
    y_test_pred = best_model.predict(X_test_scaled)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
    metrics['val_auc'] = best_auc

    return metrics


# Example usage with a dataset

def find_best_mlp(X_train, y_train, X_test, y_test, classification=True, binary = True, plot=False):
    # Combine train and validation sets
    X_combined = X_train
    y_combined = y_train

    # Standardize the data
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    X_test_scaled = scaler.transform(X_test)

    if not classification:
        y_scaler = StandardScaler()
        y_combined = y_scaler.fit_transform(y_combined.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Check if X_train has less than 3 samples
    metrics = {}
    if X_train.shape[0] <= 3:
        best_model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
        best_model.fit(X_combined_scaled, y_combined)
        y_train_pred_proba = best_model.predict_proba(X_combined_scaled)[:, 1]
        metrics['val_auc'] = roc_auc_score(y_train, y_train_pred_proba)
    else:
        # Define the hyperparameter space for random search
        param_dist = {
            'hidden_layer_sizes': [(16,), (32,), (64,), (16, 16), (32, 32), (64, 64), (16, 16, 16), (32, 32, 32), (64, 64, 64)],
            'learning_rate_init': np.logspace(-4, -2, num=5),  # Learning rates
            'alpha': np.logspace(-5, -2, num=5),  # Weight decay (L2 penalty)
            'activation': ['relu'],  # Common activations
            'solver': ['adam'],  # Solver for weight optimization
        }

        cv = get_cv(X_train)
        splits = get_splits(cv, X_combined_scaled, y_combined)

        # MLP model selection based on classification or regression
        best_score = -float('inf')
        best_params = None
        best_model = None

        # Number of random configurations to try
        n_iter = 1
        np.random.seed(42)

        for _ in range(n_iter):
            # Sample random hyperparameters
            curr_params = {
                'hidden_layer_sizes': param_dist['hidden_layer_sizes'][np.random.randint(len(param_dist['hidden_layer_sizes']))],
                'learning_rate_init': param_dist['learning_rate_init'][np.random.randint(len(param_dist['learning_rate_init']))],
                'alpha': np.random.choice(param_dist['alpha']),
                'activation': 'relu',
                'solver': 'adam'
            }

            # Cross validation scores for this parameter set
            cv_scores = []
            for train_idx, val_idx in splits:
                # Create and fit model
                model = MLPClassifier(max_iter=1000, random_state=42, **curr_params)
                model.fit(X_combined_scaled[train_idx], y_combined[train_idx])
                
                # Get probabilities and compute AUC
                val_proba = model.predict_proba(X_combined_scaled[val_idx])[:, 1]
                cv_scores.append(roc_auc_score(y_combined[val_idx], val_proba))

            # Average score across folds
            mean_cv_score = np.mean(cv_scores)

            # Update best if improved
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_params = curr_params

        metrics['val_auc'] = best_score

        # Retrain on full training data with best params
        best_model = MLPClassifier(max_iter=1000, random_state=42, **best_params)
        best_model.fit(X_combined_scaled, y_combined)

    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_scaled)

    # Calculate F1 score and accuracy for classification
    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
    return metrics





# X_train, y_train, X_test, y_test = get_xy_traintest(823, '154_athlete_sport_football', 9, num_test = 300, model_name = 'gemma-2-9b')
# # print(X_train.shape)
# start_time = time.time()

# # # metrics = find_best_xgboost(X_train, y_train, X_test, y_test)
# metrics = find_best_mlp(X_train, y_train, X_test, y_test)

# end_time = time.time()
# print(f"Time taken: {end_time - start_time:.2f} seconds")
# # print(f"scikit-learn version: {sklearn.__version__}")
# # X_train, y_train, X_test, y_test = get_xy_traintest(numbered_dataset_tag = '5_hist_fig_ismale', num_train=1024, layer = 9)
# metrics = find_best_reg(X_train, y_train, X_test, y_test, seed = 12)
# print(metrics)