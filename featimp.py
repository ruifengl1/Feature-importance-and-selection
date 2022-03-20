import numpy as np
import pandas as pd
from scipy.stats import spearmanr as _spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _pca
from pca import pca
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

# constants
fig_dpi=90

def spearCorr(X, Y):
    corr, _ =  _spearmanr(X, Y)
    data = {
    'feature_names':X.columns[np.argsort(abs(corr[:-1,-1]))[::-1]],
    'correlation': abs(corr[:-1,-1])[np.argsort(abs(corr[:-1,-1]))[::-1]]
    } 
    return pd.DataFrame(data).set_index('feature_names')


def pca_imp(X, Y):
    # standardize
    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    # perform PCA
    pca = _pca(n_components = 0.95)
    pca.fit_transform(X_std)

    pca_comp_df = pd.DataFrame(pca.components_.T, columns=['PC%s' % str(i+1) for i in range(len(pca.components_))], index = X.columns ) 

    # print explain variance ratio
    pca_ratio_df = pd.DataFrame([pca.explained_variance_ratio_], columns=['PC%s' % str(i+1) for i in range(len(pca.components_))], index=['explained_variance_ratio_'])

    data = {
    'feature_names': X.columns[np.argsort(abs(pca.components_[0]))[::-1]],
    'scores':abs(pca.components_[0])[np.argsort(abs(pca.components_[0]))[::-1]]
    }
    pca_fea_importance_df = pd.DataFrame(data).set_index('feature_names')
    return pca_comp_df, pca_ratio_df, pca_fea_importance_df

def pca_importances(X, Y):
    # standardize
    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    # perform PCA for feature importance 
    pca_fea = pca(n_components = 0.95)
    out = pca_fea.fit_transform(X_std, verbose=1)
    rank_fea_idx = [int(i)-1 for i in out['topfeat'].feature.values]
    return np.array(X.columns)[rank_fea_idx].tolist()


def pca_heatmap(df_norm):
    plt.figure(dpi=fig_dpi)
    ax_normal = sns.heatmap(df_norm, cmap="RdBu")
    ax_normal.set_title("PCA feature importance")

def drop_col_importances(X, Y):
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    rf.fit(X, Y)
    imp = dropcol_implementation(rf, X, Y)
    return imp


def dropcol_implementation(model, X_train, y_train):
    model_ = clone(model)
    model_.random_state = 999
    model_.fit(X_train, y_train)
    baseline = model_.score(X_train, y_train)
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        model_ = clone(model)
        model_.random_state = 999
        model_.fit(X_train.drop(col,axis=1), y_train)
        s = model_.score(X_train.drop(col,axis=1), y_train)
        imp.append(baseline - s)
    imp = np.array(imp)
    importance_df = pd.DataFrame(data={'feature_names':X_train.columns, 'importances':imp})
    importance_df = importance_df.set_index('feature_names')
    importance_df = importance_df.sort_values('importances', ascending=False)
    return importance_df


def permute_importances(X, Y):
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    rf.fit(X, Y)
    imp = permutation_implementation(rf, X, Y)
    return imp

def permutation_implementation(model, X_train, y_train):
    baseline = model.score(X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = model.score(X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
        
    # create df
    data = {
        'feature_names': X_train.columns[np.argsort(imp)[::-1]],
        'importances':sorted(imp, reverse=True)
    }
    importance_df = pd.DataFrame(data).set_index('feature_names')
    return importance_df


def shap_importances(X, Y):
    model = XGBRegressor().fit(X, Y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # shap values dataframe
    vals= np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns,vals)),columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    feature_importance = feature_importance.set_index('col_name')
    return shap_values, feature_importance

def validation_test(features, X, Y):
    mse_scores = []
    for i in range(len(features)):
        feas = features[:i+1]
        X_fea = X[feas]
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        model.fit(X, Y)
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        mse = cross_val_score(model, X_fea, Y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        mse_scores.append(abs(mse).mean())
    return mse_scores

def auto_selection(features, X, Y):
    x_train = X.copy()

    model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    model.fit(X, Y)
    cv = KFold(n_splits=3, random_state=1, shuffle=True)
    mse_scores = cross_val_score(model, x_train, Y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    prev_mse = abs(mse_scores).mean()
    
    while len(x_train.columns) != 0:
        print(f'Previous feature MSE: {round(prev_mse,3)}')
        # drop the least important feature
        fea_name = features[-1]
        x_train = x_train.drop([fea_name], axis=1)
        print(f'Drop the least important feature: {fea_name}')

        # compute mse
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        model.fit(X, Y)
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        mse_scores = cross_val_score(model, x_train, Y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        mse = abs(mse_scores).mean()
        print(f'Re-compute MSE: {round(mse,3)}')

        # check mae
        if mse >prev_mse:
            col_names = list(x_train.columns)
            col_names.append(fea_name)
            return col_names

        prev_mse = mse

        # recompute feature importance
        _, shap_imp = shap_importances(x_train, Y)
        features = shap_imp.index.values
#         print('feature importance: ', shap_imp)

    return None















