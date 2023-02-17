
import h5py
import numpy as np
import pickle
import random
import plotly.graph_objects as go
import pandas as pd
import sknetwork as skn
import seaborn as sns

import matplotlib.pyplot as plt

import plotly.io as pio
pio.renderers.default = 'svg'


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sknetwork.clustering import Louvain
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph

#to create Adjacency matrix for  Louvain clustering
from sklearn.metrics import pairwise_distances 
from sklearn.preprocessing import MinMaxScaler

#Evaluation and Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, v_measure_score
from sklearn.model_selection import KFold, train_test_split

#import sklearn.metrics as metrics
#import sklearn.cluster as cluster

def calculate_percent(sub_df, attrib):
    cnt = sub_df[attrib].count()
    output_sub_df = sub_df.groupby(attrib).count()
    return (output_sub_df/cnt)

def TestLabel(x):
   if x == 'pge_pca':
       return test_label_pge_pca, test_data_pge_pca
   elif x == 'resnet50_pca':
       return test_label_resnet50_pca,test_data_resnet50_pca
   elif x == 'inception3_pca':
       return test_label_inception3_pca,test_data_inception3_pca
   elif x == 'vgg16_pca':
       return test_label_vgg16_pca,test_data_vgg16_pca
   
   elif x == 'pge_umap':
       return test_label_pge_umap,test_data_pge_umap
   elif x == 'resnet50_umap':
       return test_label_resnet50_umap,test_data_resnet50_umap
   elif x == 'inception3_umap':
       return test_label_inceptionv3_umap,test_data_inceptionv3_umap
   elif x == 'vgg16_umap':
       return test_label_vgg16_umap,test_data_vgg16_umap
   
def Model_Training(i): #i=PCA/UMAP FEATURE feature_umap/pca
    [test_label, test_data] = TestLabel(i) #type test label 
    
    #kmeans
    kmeans_model = KMeans(n_clusters = 2, init = 'k-means++') #GaussianMixture(), AgglomerativeClustering(), Louvain
    kmeans_assignment = kmeans_model.fit_predict(test_data)
    
    #print("Kmeans assignment:", kmeans_assignment)
    
    #Evaluation and visualization 
    #Kmeans_model plot with centroids
    u_labels = np.unique(kmeans_assignment)
    print(u_labels)   
    centroids = kmeans_model.cluster_centers_    #centroids
    #u_labels = np.unique(kmeans_assignment)
    #print('..........................................')
    #print(u_labels)
         
    for i in u_labels:
        sns.scatterplot(test_data[kmeans_assignment == i , 0] , test_data[kmeans_assignment == i , 1])
        #plt.scatter(test_data[kmeans_assignment == i , 0] , test_data[kmeans_assignment == i , 1])
    # plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.title('kmeans')
    plt.show()       
    
  
    
    # print(kmeans_model.inertia_) # The lowest SSE value
    # print(kmeans_model.cluster_centers_) # Final locations of the centroid
    print('Number of iterations =', kmeans_model.n_iter_) # The number of iterations required to converge
    
    #louvain
    louvain_model = Louvain(resolution = 1.05, modularity = 'Newman',random_state = 0) 
    adjacency_matrix = MinMaxScaler().fit_transform(-pairwise_distances(test_data))
    sprs_matrix = csr_matrix(adjacency_matrix) #convert to sparse matrix
    louvain_assignment = louvain_model.fit_transform(sprs_matrix)
    u_labels = np.unique(louvain_assignment)
    print("louvain",u_labels)
    
    for i in u_labels:
        plt.scatter(test_data[louvain_assignment == i , 0] , test_data[louvain_assignment == i , 1])
    plt.legend()
    plt.title('louvain')
    plt.show()   
    
    #Evaluation
    print('Number of clusters from KMeans: %d and from Louvain: %d'%(np.unique(kmeans_assignment).shape[0],np.unique(louvain_assignment).shape[0]))
    kmeans_counts = np.unique(kmeans_assignment, return_counts = True)
    louvain_counts = np.unique(louvain_assignment, return_counts = True)
    print('Kmeans assignment counts')
    print(pd.DataFrame({'Cluster Index': kmeans_counts[0], 'Number of members':kmeans_counts[1]}).set_index('Cluster Index'))
    print('Louvain assignment counts')
    print(pd.DataFrame({'Cluster Index': louvain_counts[0], 'Number of members':louvain_counts[1]}).set_index('Cluster Index'))

    kmeans_silhouette = silhouette_score(test_data, kmeans_assignment)
    louvain_silhouette = silhouette_score(test_data, louvain_assignment)
    kmeans_v_measure = v_measure_score(test_label, kmeans_assignment)
    louvain_v_measure = v_measure_score(test_label, louvain_assignment)
    print(pd.DataFrame({'Metrics': ['silhouette', 'V-measure'], 'Kmeans': [kmeans_silhouette, kmeans_v_measure], 'Louvain':[louvain_silhouette, louvain_v_measure]}).set_index('Metrics'))
    
    resulted_cluster_df = pd.DataFrame({'clusterID': kmeans_assignment, 'type': test_label})
    label_proportion_df = resulted_cluster_df.groupby(['clusterID']).apply(lambda x: calculate_percent(x,'type')).rename(columns={'clusterID':'type_occurrence_percentage'}).reset_index()
    pivoted_label_proportion_df = pd.pivot_table(label_proportion_df, index = 'clusterID', columns = 'type', values = 'type_occurrence_percentage')
    
    
    f, axes = plt.subplots(1, 2, figsize=(20,5))
    number_of_tile_df = resulted_cluster_df.groupby('clusterID')['type'].count().reset_index().rename(columns={'type':'number_of_tile'})
    df_idx = pivoted_label_proportion_df.index
    (pivoted_label_proportion_df*100).loc[df_idx].plot.bar(stacked=True, ax = axes[0] )
    #print(number_of_tile_df)
    
    axes[0].set_ylabel('Percentage of tissue type')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Cluster configuration by Kmeans')
    
    resulted_cluster_df = pd.DataFrame({'clusterID': louvain_assignment, 'type': test_label})
    label_proportion_df = resulted_cluster_df.groupby(['clusterID']).apply(lambda x: calculate_percent(x,'type')).rename(columns={'clusterID':'type_occurrence_percentage'}).reset_index()
    pivoted_label_proportion_df = pd.pivot_table(label_proportion_df, index = 'clusterID', columns = 'type', values = 'type_occurrence_percentage')
    
    
    number_of_tile_df = resulted_cluster_df.groupby('clusterID')['type'].count().reset_index().rename(columns={'type':'number_of_tile'})
    df_idx = pivoted_label_proportion_df.index
    (pivoted_label_proportion_df*100).loc[df_idx].plot.bar(stacked=True, ax = axes[1] )
    
    axes[1].set_ylabel('Percentage of tissue type')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Cluster configuration by Louvain')
    f.show()
    # f.savefig('demo.png', bbox_inches='tight')
    plt.show()


    return

def Model_Evaluation(i): #i=PCA/UMAP FEATURE feature_umap/pca
    [test_label, test_data] = TestLabel(i) #type test label
    
    #kmeans
    print('----Kmeans----')
    
    #elbow method
   
    
    kmeans_silhouette = np.array([])
    kmeans_v_measure = np.array([])
    number_of_clusters = np.array([])
    wss = [] #within cluster sum of squared
    K = range (2,10)
    
    for x in K:
        kmeans_model = KMeans(n_clusters = x, init = 'k-means++') #GaussianMixture(), AgglomerativeClustering(), Louvain
        kmeans_assignment = kmeans_model.fit_predict(test_data)
        
        wss_iter = kmeans_model.inertia_
        wss.append(wss_iter)
        
        kmeans_silhouette_append = np.array([silhouette_score(test_data, kmeans_assignment)])
        kmeans_silhouette = np.append(kmeans_silhouette, kmeans_silhouette_append)
        
        kmeans_v_measure_append = v_measure_score(test_label, kmeans_assignment)
        kmeans_v_measure = np.append(kmeans_v_measure, kmeans_v_measure_append)
        
        number_of_clusters = np.append(number_of_clusters, np.array([x]))
        
    mycenters = pd.DataFrame({'Clusters': K, 'WSS': wss})
    print(mycenters)
    
    #sns.lineplot(x = 'Clusters', y = 'WSS', data = mycenters, marker = '*' )
    plt.plot(K, wss,'r*-.')
    plt.title('Elbow')
    plt.xlabel('Clusters')
    plt.ylabel('WSS')
    plt.show()
    #silhoutte values *****
    #for i in range(2, 13):
    #    labels = cluster.KMeans(n_clusters=i, init = 'k-means++', random_state=200).fit(test_data).labels_
    #    print("Silhouette score for k(clusters): "+str(i)+"is "+str(metrics.silhouette_score(test_data, labels,metric="euclidean", sample_size = 1000, random_state=200)) )
    #
    
    plt.plot(number_of_clusters, kmeans_silhouette,'r*-.')
    plt.title('Silhouette Plot')
    plt.show()
    index_max = np.argmax(kmeans_silhouette)
    best_number_of_clusters = number_of_clusters[index_max]
    print('Number of clusters (highest silhouette score) =',best_number_of_clusters)

    plt.plot(number_of_clusters, kmeans_v_measure,'b*-.')
    plt.title('V measure Plot')
    plt.show()
    index_max = np.argmax(kmeans_v_measure)
    best_number_of_clusters = number_of_clusters[index_max]
    print('Number of clusters (highest v_measure score) =',best_number_of_clusters)
    
    #Find peaks
    index_max = np.array([])
    for x in range (0,kmeans_v_measure.size-1):
        val1 = kmeans_v_measure[x]
        val2 = kmeans_v_measure[x+1]
        if(val2-val1 < 0):
            index_max = np.append(index_max, x)

    for x in index_max:
        cluster_no = number_of_clusters[int(x)]
        print('Peaks_cluster_no (v_measure) =',cluster_no)
    
    #louvain
    
    print('----Louvain----')
    louvain_silhouette = np.array([])
    louvain_v_measure = np.array([])
    louvain_resolution = np.array([])

    z = 0
    for x in range (90,120,1):
        if(z<=9):
            y = x/100
            louvain_model = Louvain(resolution = y, modularity = 'Newman',random_state = 0) 
            adjacency_matrix = MinMaxScaler().fit_transform(-pairwise_distances(test_data))
            sprs_matrix = csr_matrix(adjacency_matrix) #convert to sparse matrix
            louvain_assignment = louvain_model.fit_transform(sprs_matrix)
            # if (np.unique(louvain_assignment).shape[0]<=9):
        
            louvain_silhouette_append = silhouette_score(test_data, louvain_assignment)
            louvain_silhouette = np.append(louvain_silhouette, louvain_silhouette_append)
            
            louvain_v_measure_append = v_measure_score(test_label, louvain_assignment)
            louvain_v_measure = np.append(louvain_v_measure, louvain_v_measure_append)
            
            louvain_resolution = np.append(louvain_resolution, np.array([y]))
            z= np.unique(louvain_assignment).shape[0] #limits max cluster size to 9
    
    # print(np.unique(louvain_assignment).shape[0])
    print("Louvain resoultions",louvain_resolution)
    plt.plot(louvain_resolution, louvain_silhouette,'r*-.')
    plt.title('Silhouette Plot')
    plt.show()
    index_max = np.argmax(louvain_silhouette)
    best_resolution = louvain_resolution[index_max]
    print('Resolution (highest silhouette score) =',best_resolution )
    
    plt.plot(louvain_resolution, louvain_v_measure,'b*-.')
    plt.title('V measure Plot')
    plt.show()
    
    #Find peaks
    index_max = np.array([])
    for x in range (0,louvain_v_measure.size-1):
        val1 = louvain_v_measure[x]
        val2 = louvain_v_measure[x+1]
        if(val2-val1 < 0):
            index_max = np.append(index_max, x)

    for x in index_max:
        resolution = louvain_resolution[int(x)]
        print('Peak values (v_measure plot) =',resolution)
 
    return
    
# EXPLORATORY ANALYSIS
def Exploratory_Analysis(i): #i=PCA/UMAP FEATURE feature_umap/pca



    [test_label, test_data] = TestLabel(i) #type test label
    
    # print(test_label, test_data)
        
    traces = []
    for name in np.unique(labels):
        trace = go.Scatter3d(
            x=test_data[test_label==name,0],
            y=test_data[test_label==name,1],
            z=test_data[test_label==name,2],
            mode='markers',
            name=name,
            marker=go.scatter3d.Marker(
                size=4,
                opacity=0.8
            )
    
        )
        traces.append(trace)
    
    
    data = go.Data(traces)
    layout = go.Layout(
                showlegend=True,
        scene=go.Scene(
                    xaxis=go.layout.scene.XAxis(title='PC1'),
                    yaxis=go.layout.scene.YAxis(title='PC2'),
                    zaxis=go.layout.scene.ZAxis(title='PC3')
                    )
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        title="First 3 pricipal components of PathologyGAN's PCA feature",
        legend_title="Legend Title",
    )
    
    fig.show()
    return
if __name__ == "__main__":    
    pge_path = 'colon_nct_feature/pge_dim_reduced_feature.h5'
    resnet50_path = 'colon_nct_feature/resnet50_dim_reduced_feature.h5'
    inceptionv3_path = 'colon_nct_feature/inceptionv3_dim_reduced_feature.h5'
    vgg16_path = 'colon_nct_feature/vgg16_dim_reduced_feature.h5'
    
    pge_content = h5py.File(pge_path, mode='r')
    resnet50_content = h5py.File(resnet50_path, mode='r')
    inceptionv3_content = h5py.File(inceptionv3_path, mode='r')
    vgg16_content = h5py.File(vgg16_path, mode='r')
    
    #PCA feature from 4 feature sets: pge_latent, resnet50_latent, inceptionv3_latent, vgg16_latent
    pge_pca_feature  = pge_content['pca_feature'][...]
    resnet50_pca_feature  = resnet50_content['pca_feature'][...]
    inceptionv3_pca_feature = inceptionv3_content['pca_feature'][...]
    vgg16_pca_feature  = vgg16_content['pca_feature'][...]
    
    #UMAP feature from 4 feature sets: pge_latent, resnet50_latent, inceptionv3_latent, vgg16_latent
    pge_umap_feature  = pge_content['umap_feature'][...]
    resnet50_umap_feature = resnet50_content['umap_feature'][...]
    inceptionv3_umap_feature  = inceptionv3_content['umap_feature'][...]
    vgg16_umap_feature  = vgg16_content['umap_feature'][...]
    
    #tissue type as available ground-truth: labels
    filename  = np.squeeze(pge_content['file_name'])
    labels = np.array([x.split('/')[2] for x in filename])
    
    #TO SELECT RANDOM 200
    
    random.seed(0)
    selected_index_pge_pca = random.sample(list(np.arange(len(pge_pca_feature))), 200) #picks 200 data
    selected_index_resnet50_pca = random.sample(list(np.arange(len(resnet50_pca_feature))), 200)
    selected_index_inception3_pca = random.sample(list(np.arange(len(inceptionv3_pca_feature))), 200)
    selected_index_vgg16_pca = random.sample(list(np.arange(len(vgg16_pca_feature))), 200)
    
    selected_index_pge_umap = random.sample(list(np.arange(len(pge_umap_feature))), 200) #picks 200 data
    selected_index_resnet50_umap = random.sample(list(np.arange(len(resnet50_umap_feature))), 200)
    selected_index_inceptionv3_umap = random.sample(list(np.arange(len(inceptionv3_umap_feature))), 200)
    selected_index_vgg16_umap = random.sample(list(np.arange(len(vgg16_umap_feature))), 200)
    
    test_data_pge_pca = pge_pca_feature[selected_index_pge_pca] #writes into array
    test_label_pge_pca = labels[selected_index_pge_pca]
    test_data_resnet50_pca = pge_pca_feature[selected_index_resnet50_pca]
    test_label_resnet50_pca = labels[selected_index_resnet50_pca]
    test_data_inception3_pca = pge_pca_feature[selected_index_inception3_pca]
    test_label_inception3_pca = labels[selected_index_inception3_pca]
    test_data_vgg16_pca = pge_pca_feature[selected_index_vgg16_pca]
    test_label_vgg16_pca = labels[selected_index_vgg16_pca]
    
    test_data_pge_umap = pge_pca_feature[selected_index_pge_umap]
    test_label_pge_umap = labels[selected_index_pge_umap]
    test_data_resnet50_umap = pge_pca_feature[selected_index_resnet50_umap]
    test_label_resnet50_umap = labels[selected_index_resnet50_umap]
    test_data_inceptionv3_umap = pge_pca_feature[selected_index_inceptionv3_umap]
    test_label_inceptionv3_umap = labels[selected_index_inceptionv3_umap]
    test_data_vgg16_umap = pge_pca_feature[selected_index_vgg16_umap]
    test_label_vgg16_umap = labels[selected_index_vgg16_umap]

#pge, resnet50, inceptionv3, vgg16 + _pca/umap
    #Exploratory_Analysis('pge_umap')
    Model_Training('pge_umap')
    Model_Evaluation('pge_umap')
    #Exploratory_Analysis('pge_pca')
    #Model_Training('pge_pca')
    #Model_Evaluation('pge_pca')
    
    #Exploratory_Analysis('pge_umap')
    #Model_Training('resnet50_umap')
    #Model_Evaluation('resnet50_umap')
    #Exploratory_Analysis('pge_pca')
    #Model_Training('resnet50_pca')
    
    #Model_Evaluation('resnet50_pca')
    
    