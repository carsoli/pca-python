#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../Assignment1'))
	print(os.getcwd())
except:
	pass

#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
def read_sv(fpath, sep=',', hdr=None):
    df = pd.read_csv(fpath, sep=sep, header=hdr)
    return df


#%%
def plot_data_pca(st_data, evecs, evals, name, dpi):
    x1, x2 = zip(*st_data) #x1, x2 = st_data.transpose() 
    plt.scatter(x1, x2, s=20, c='m', marker='o', alpha=0.4)
    mu = st_data.mean(axis=0)
    #evals[:] multplies each pca with its corr. lambda
    plt.quiver(mu[0], mu[1], evals[:]*evecs[:][0], evals[:]*evecs[:][1], width=0.004, 
               scale=1.0, color='black', label="Principal Components")
    plt.legend()
    plt.axis('equal')
    plt.savefig(name, dpi=dpi, bbox_inches='tight')
    plt.show()


#%%
def PCA(XS, k=1): # no Y's for this ds
    means = np.mean(XS, axis=0) #1D-array(vector) of size = n (features' number)
    N = XS.shape[0]-1 #b/c cov runs from 1 to n 
    cov = (1/N)*( ((XS - means).transpose()).dot(XS - means) ) #(nxn matrix) 
    #cov_check = np.cov(XS.transpose()) #covariance matrix
    evals, ematrices = LA.eig(cov)
    
    evecs = ematrices.transpose()
    nvals = -evals
    indices = np.argsort(nvals)
    nvals.sort()
    
    evecs_sorted = []
    for i in indices:
        evecs_sorted.append(evecs[i])

    projections = pd.DataFrame()
    for idx in range(0,k): #project data on one PC at a time
        s = pd.DataFrame({"PCA"+str(idx+1): XS@evecs_sorted[idx]})
        projections = pd.concat([projections,s],axis=1)
        
    projected_data = XS@evecs_sorted
    return (evecs_sorted, -nvals, projected_data, projections)


#%%
def plot_2overlay_hists(x, lx, cx, y, ly, cy, bins, op, name):
    plt.hist( x, bins, alpha=op, label=lx, color=cx )
    plt.hist( y, bins, alpha=op, label=ly, color=cy )
    plt.legend(loc='upper right')
    plt.axis('tight')
    plt.title(name.rpartition(".")[0], loc='center')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()


#%%
def plot_differences(pairs, name):
    x = [p[1] for p in pairs]
    y = [p[0] for p in pairs]
    plt.plot(x, y)
    plt.xlabel("PCA Index")
    plt.ylabel("Distance b/w Means of Home & Away Wins of Projected Data")
    plt.axis('tight')
    plt.title(name.rpartition(".")[0], loc='center')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()


#%%
def task1():
    df = read_sv('Data.txt', '\s+', None)
    D = df.shape[1] #number of features, before reduction
    X = df.values
    X_standardized = StandardScaler().fit_transform(X)
    XS = X_standardized
    evecs, evals, projected_data, projections = PCA(X,2)
    plot_data_pca(XS, evecs, evals, 'Data_PCA.png', 300) 


#%%
def task2():
    df = pd.read_excel('EPL.xlsx')
    features = list(df.columns)
    features.remove('HomeTeam')
    features.remove('AwayTeam')
    features.remove('FTR')
    labels = 'FTR'
    Y = df.loc[:, labels] 
    X = df.loc[:, features]
    X_standardized = StandardScaler().fit_transform(X)
    XS = X_standardized
    
    # PCA on data
    evecs, evals, projected_data, projections = PCA(X, 8)
    # PCA on standardized_data
    evecs_st, evals_st, projected_data_st, projections_st = PCA(XS, 8)

    bins = 10
    opacity = 0.5
    YS = pd.Series(Y)
    mask = (df.FTR == 'H')
    dist_idx_pairs= []
    dip = dist_idx_pairs
    dist_idx_pairs_st = []
    dip_st = dist_idx_pairs_st
    
    for idx in range(0, projections.shape[1]):
        col_name = projections.columns[idx]

        df = pd.concat([projections[col_name], YS], axis=1)
        df_st = pd.concat([projections_st[col_name], YS], axis=1)

        home_projections = df[mask][col_name].values
        home_projections_st = df_st[mask][col_name].values
            
        away_projections = df[~mask][col_name].values
        away_projections_st = df_st[~mask][col_name].values

        plot_2overlay_hists(home_projections, 'Home', 'red', 
                            away_projections, 'Away', 'blue', 
                            bins, opacity, "Proj_" + col_name + ".png")
        
        plot_2overlay_hists(home_projections_st, 'Home', 'red', 
                    away_projections_st, 'Away', 'blue', 
                    bins, opacity, "StProj_" + col_name + ".png")
        
        mu_home = home_projections.mean(axis=0)
        mu_away = away_projections.mean(axis=0)
        dip.append( (abs(mu_home - mu_away), (idx+1)) )

        mu_home_st = home_projections_st.mean(axis=0)
        mu_away_st = away_projections_st.mean(axis=0)
        dip_st.append( (abs(mu_home_st - mu_away_st), (idx+1)) )


    plot_differences(dip, "Distance.png")
    plot_differences(dip_st, "StDistance.png")


#%%
def main():
    task1()
    task2()

main()


