# For modellling
from tensorflow.keras.utils import load_img 
from keras.applications.vgg16 import preprocess_input 
# from PIL import Image 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
# from tensorflow.keras.preprocessing.image import img_to_array

# other packages
import pandas as pd
import numpy as np
# from numpy import random
from helpfile import *
import plotly.express as px
import os
import io 
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
os.chdir(r"C:\Users\rinar\Documents\Master_Thesis\Project_Master_Thesis")

#### VGG16 ###
# Model functions 
# load the model first and pass as an argument
model_VGG16 = VGG16(weights="C:/Users/rinar/Anaconda3/envs/myEnvironmentThesis/Lib/site-packages/keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
model_VGG16 = Model(inputs = model_VGG16.inputs, outputs = model_VGG16.layers[-2].output)

# Data 
data = pd.read_csv("data/data_clustered.csv")

# data = data[['id', 'lon', 'lat', 'moll_x', 'moll_y', 'country','land_cover_name', 'closestTurbine', 'distance', 'WFid_avg_min0.8']]
# random.seed(123)
# sample_id = random.sample(sorted(np.unique(data["WFid"])[1::]), 1000)
# get farms with more than 4 turbines
wfid_count = data["WFid"].value_counts()
wfid_count = wfid_count.to_frame()
ind_more_than4 = wfid_count[wfid_count.WFid>5].index.values
len(ind_more_than4)
df_more_than4= data[data["WFid"].isin(ind_more_than4)]
data_shapecluster = df_more_than4[~df_more_than4["WFid"].isin([-1])]
sample_id = np.unique(data_shapecluster["WFid"])
sample_wfid = sample_id
len(sample_wfid)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# if not os.path.exists("images"):
#     os.mkdir("images")
def create_figures(X, wf_ids): # Returns figures list 
    figs = []
    for wid in wf_ids:
        ##
        # wid = sample_wfid[4]
        # wf = data[data["WFid"] == wid]
        ##
        wf = X[X["WFid"] == wid]
        wf_proj = proj_to_wf_center(wf)*1000
    
        fig = px.scatter(wf_proj, x = "x", y = "y")
        
        fig = go.Figure(fig)
        fig.update_traces(
            marker_size=5,
            marker=dict(color="black"),
            marker_line=dict(width=1, color="black"),
            selector=dict(mode='markers')
        )

        fig.update_xaxes(showgrid=True, 
            dtick = 100,
            range=[-5000, 5000], 
            zeroline = True, 
            showline = True,
            # visible=False, 
            showticklabels=False)
            
        fig.update_yaxes(showgrid=True, 
                    dtick = 100,
                    range=[-4000, 4000], 
                    zeroline = True, 
                    showline = True,
                    # visible=False, 
                    showticklabels=False)        
        fig.update_layout(
            height = 500, 
            width = 500
        )
        # fig.show()
        figs.append(fig)
    
    return figs

def extract_features(fig, model):
    # load the image as a 224x224 array
    fig_bytes = fig.to_image(format="png" )
    buf = io.BytesIO(fig_bytes)
    img = load_img(buf, target_size=(224,224))    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
    
def formatize_features(figs, sample_wfid, model):
    data = {}
    p =  r"C:\Users\rinar\Documents\Master_Thesis\Project\Model_Development\windfarm_features.pkl"
    # loop through each image in the dataset
    for wfid_name, windfarm in zip(sample_wfid, figs):
        # try to extract the features and update the dictionary
        feat = extract_features(windfarm,model)
        data[wfid_name] = feat
       
    # get a list of the filenames
    # filenames = np.array(list(data.keys()))
    # get a list of just the features
    feat = np.array(list(data.values()))
    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1,4096)
    return feat

# Function to Extract features from the images
# def image_feature(figs, model):
#     # model = InceptionV3(weights='imagenet', include_top=False)
#     features = [];
#     for fig in figs:
#         fig_bytes = fig.to_image(format="png" )
#         buf = io.BytesIO(fig_bytes)
#         img = load_img(buf, target_size=(224,224)) 
#         x = img_to_array(img)
#         x=np.expand_dims(x,axis=0)
#         x=preprocess_input(x)
#         feat=model.predict(x)
#         feat=feat.flatten()
#         features.append(feat)
#     return features
    
def PCA_analyis(feat, n_components):
    pca = PCA(n_components=n_components, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)
    return x

def eval_table(feats):
    # feats = feats_vgg16_old
    ncomps = [ 2, 3, 5, 6, 10, 20, 40, 50, 100, 200]
    ks = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    eval_elbow = pd.DataFrame(columns = ncomps, index = ks)
    eval_shscore = pd.DataFrame(columns = ncomps, index = ks)
    eval_chscore = pd.DataFrame(columns = ncomps, index = ks)
    for ncomp in ncomps:
        # ncomp = 2
        x = PCA_analyis(feats, ncomp)
        for k in ks:
            # k = 2
            km = KMeans(n_clusters=k, random_state=22)
            km.fit(x)
            labels = km.labels_
            # len(labels)
            eval_elbow.loc[k, ncomp] = km.inertia_
            eval_shscore.loc[k, ncomp] =silhouette_score(x, labels)
            eval_chscore.loc[k, ncomp] =calinski_harabasz_score(x, labels)
            
    return eval_elbow, eval_shscore, eval_chscore

def k_clusters_elbow(x):
        ks = pd.Series(range(1,15))
        eval_elbow = pd.DataFrame(columns = ["100"], index = ks)
        for k in ks:
            # k = 2
            km = KMeans(n_clusters=k, random_state=22)
            km.fit(x)
            # len(labels)
            eval_elbow.loc[k, "100"] = km.inertia_
        
        
        fig_elbow = px.line(x =eval_elbow.index, y=eval_elbow["100"])
        fig_elbow.show()

# def model_predict_eval(feats, n_components):
#     x = PCA_analyis(feats, n_components)

#     sse = []
#     labels_list = []
#     list_k = list(range(2, 30))
#     for k in list_k:
#         km = KMeans(n_clusters=k, random_state=22)
#         km.fit(x)
#         labels = km.labels_
#         labels_list.append(labels)
#         sse.append(km.inertia_)

#     sh_scores = []
#     for lab in labels_list[1:]:
#         sh_scores.append(silhouette_score(x, lab)) 

#     ch_scores = []
#     for lab in labels_list[1:]:
#         ch_scores.append(calinski_harabasz_score(x, lab)) 

#     fig_elbow= px.line(x =list_k, y=sse, title="Elbow Method")
#     fig_elbow.show()

#     fig_sh= px.line(x =list_k[1:], y=sh_scores, title = "Silhouette Score" )
#     fig_sh.show()

#     fig_sh= px.line(x =list_k[1:], y=ch_scores, title = "Calinski_Harabasz_score")
#     fig_sh.show()

#     return [list_k, sse , sh_scores, ch_scores]


# fig_bytes = fig.to_image(format="png" )
#     buf = io.BytesIO(fig_bytes)
#     img = load_img(buf, target_size=(224,224))    # convert from 'PIL.Image.Image' to numpy array
#     img = np.array(img) 


# figs[0]
start = time.time()
figs = create_figures(data_shapecluster, sample_wfid)
feats_vgg16 = pd.DataFrame(formatize_features(figs, sample_wfid, model_VGG16))
feats_vgg16.to_csv("data/features_vgg16_new_test.csv")
feats_vgg16 = pd.read_csv("data/features_vgg16_new_test.csv")


see_vgg16, silhouette_vgg16, calinski_vgg16 = eval_table(feats_vgg16)

see_vgg16.to_csv("tables/see_vgg16_new.csv")
silhouette_vgg16.to_csv("tables/silhouette_vgg16_newimg_new.csv")
calinski_vgg16.to_csv("tables/calinski_newimg_new.csv")

end = time.time()
print(end - start)

# silhouette_vgg16_old_oldImg = pd.read_csv("tables\silhouette_vgg16_full.csv").iloc[: , 1:]
# silhouette_vgg16_old = pd.read_csv("tables\silhouette_vgg16_oldProcess_newimg.csv").iloc[: , 1:]
# silhouette_inception = pd.read_csv("tables\silhouette_inception_newimg.csv").iloc[: , 1:]
# silhouette_vgg16 = pd.read_csv("tables\silhouette_vgg16_newimg.csv").iloc[: , 1:]

def plot_elbow(tab):
    components_list=list(tab.columns)
    tab_melt = pd.melt(tab, value_vars=components_list,value_name='see_score', ignore_index=False)
    fig_elbow= px.line(x =tab_melt.index, y=tab_melt.see_score,color = tab_melt.variable,  title="Elbow Method")
    fig_elbow.show()
    return fig_elbow

def plot_silhouette(tab):
    components_list=list(tab.columns)
    tab_melt = pd.melt(tab, value_vars=components_list,value_name='s_score', ignore_index=False)
    fig_silhouette= px.line(x =tab_melt.index, y=tab_melt.s_score,color = tab_melt.variable,  title="Silhouette Score")
    fig_silhouette.show()
    return fig_silhouette

def plot_calinski(tab):
    components_list=list(tab.columns)
    tab_melt = pd.melt(tab, value_vars=components_list,value_name='ch_score', ignore_index=False)
    fig_calinski= px.line(x =tab_melt.index, y=tab_melt.ch_score,color = tab_melt.variable,  title="Calinski Harabasz Score")
    fig_calinski.show()
    return fig_calinski

plot_elbow(see_vgg16)
plot_silhouette(silhouette_vgg16)
plot_calinski(calinski_vgg16)

# x = PCA_analyis(feats_vgg16, 2)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples



pca = PCA(n_components=2,  whiten=False)


pca = PCA().fit(feats_vgg16)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.xlim([2, 200])
plt.show()


# pca_nw = PCA(n_components=2,  whiten=False)
x = pd.DataFrame(pca.fit_transform(feats_vgg16))
# x.loc[:,['Land Cover', 'Landform', 'Elevation', 'Turbine Spacing']] = data.loc[:,['Land Cover', 'Landform', 'Elevation', 'Turbine Spacing']]
# x_nw = pca_nw.fit_transform(feats_vgg16)
x = PCA_analyis(feats_vgg16, 20)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# from sklearn.model_selection import train_test_split
# x["shape"] = None
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size==5000, shuffle=True)

# k_clusters_elbow(x)
# n_clust 11, random state 658
kmeans = KMeans(n_clusters=3, random_state=658)
kmeans.fit(x)
labels = kmeans.labels_
len(labels)


# kmeans_nw = KMeans(n_clusters=6, random_state=1)
# kmeans_nw.fit(x_nw)
labels_nw = labels


def divisors(n):
    result = []
    for i in range(1, n//2 + 1):
        if n % i == 0:
            result.append(i)
    result.append(n)
    return result

# Figure of groups 
def create_groups_figure(df, group_labels, wf_ids):
    group_fig_list = []

    # group_labels = labels
    # wf_ids = sample_wfid
    # df = pd.read_csv("data/clustered_data.csv")
    unique_labels = np.unique(group_labels)
    for shape_group in unique_labels:   
        # shape_group = unique_labels[0]
        bool_index_isingroup = group_labels == shape_group
        int_index_isingroup = [i for i, val in enumerate(bool_index_isingroup) if val]
        wfid_index_isingroup =  [wf_ids[i] for i in int_index_isingroup]
        
        # Get column and row divisors and set them
        n_wfs_ingroup = len(wfid_index_isingroup)
        if (n_wfs_ingroup >100):
            plot_cols = 10
            plot_rows = 10  
        else: 
            divisors_lst = divisors(int(n_wfs_ingroup))
            if len(divisors_lst) ==2:
                wfid_index_isingroup.remove(wfid_index_isingroup[0])
                # Get column and row divisors and set them
                n_wfs_ingroup = len(wfid_index_isingroup)
                divisors_lst = divisors(n_wfs_ingroup)
            # figs_ingroup = figs[wfid_index_isingroup]
            plot_cols = divisors_lst[round(len(divisors_lst)/2)]
            plot_rows = int(n_wfs_ingroup/plot_cols)
        
        
        # Data only containing wfs in shape clustering group
        data_group = df[df["WFid"].isin(wfid_index_isingroup)]

        fig = make_subplots(rows=plot_rows, cols=plot_cols) #horizontal_spacing = 0.01,vertical_spacing = 0.02)
        # fig = make_subplots(rows=1, cols=2) #horizontal_spacing = 0.01,vertical_spacing = 0.02)

        x = 0 
        for i in range(1, plot_rows + 1):
            for j in range(1, plot_cols + 1):
                # #update common attributes:
                # if x == 169 or x > (len(wfid_index_isingroup)-1):
                #     next
                # Get wind farm through id
                current_wfid = wfid_index_isingroup[x]
                current_wf = data_group[data_group["WFid"] == current_wfid]
                proj_wf = proj_to_wf_center(current_wf)*1000
                

              
                # wffig = px.scatter(x = proj_wf["x"], y =  proj_wf["y"])
                
                # wffig.data[0]
                fig.add_trace(go.Scatter(
                        x=proj_wf["x"], 
                        y=proj_wf["y"], 
                        mode='markers', 
                        marker=dict(size=5,color="black")),
                
                    # go.Scatter(
                    #     mode= "marker",
                    #     x=proj_wf["x"],
                    #     y=proj_wf["y"], )
                        
                    row=i, col=j)
                

                x=x+1


        fig.update_xaxes(range=[-5000, 5000])
        fig.update_yaxes(range=[-5000, 5000])
        fig.update_layout(height = 5000, width = 5000)
        fig.show()

        group_fig_list.append(fig)



    return group_fig_list 

# df = pd.read_csv("data/clustered_data.csv")

# wf_ids = sample_wfid.reset_index()["WFid"]#[0:1000]
res  = create_groups_figure(data_shapecluster, labels, sample_wfid)       
# len(sample_wfid)
# sample_wfid= data_shapecluster.id.to_list()
# res  = create_groups_figure(data_shapecluster, labels, sample_wfid)       
# len(labels)

# res_nw  = create_groups_figure(data, labels_nw, sample_wfid)      


# Write predicted shape labels 
labels_towrite = pd.DataFrame()
labels_towrite["WFid"] = sample_wfid#np.unique(data.WFid)
labels_towrite["Shape"] = labels
labels_towrite.to_csv("data/data_enrichment/WFshapes_pred.csv")



# df["shape"] = labels
# aggregate_data["shape"] = np.insert(group_labels, 0, -1)#[[-1, labels]]
# aggregate_data.loc[~index_filtered_wfs, "shape"] = "3 turbines"
# aggregate_data.loc[index_filtered_wfs, "shape"] = group_labels
# aggregate_data.loc[(aggregate_data["WFid"] == -1), "shape"] = -1

# aggregate_data.iloc[:, 2::].to_csv("data/aggregate_wf_data.csv")