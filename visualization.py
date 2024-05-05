import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold._t_sne import TSNE

from data_utils import getting_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import math

import spacy

def create_graph(_reverse_mapper: dict):
   """
    This function is responsible for creating a graph from the reverse mapper
    :param _reverse_mapper: The reverse mapper that we have
    :return:
   """
   PERCENTAGE = 0.00001
   graph = nx.DiGraph()
   # We want to add only 1% edges to the graph, but we want to make sure that we have at least 1 edge for each node
   for key_url, urls in _reverse_mapper.items():
       graph.add_node(key_url)
       max_number_of_edges = max(1, int(len(urls) * PERCENTAGE))
       counter = 0
       for url in urls:
          graph.add_edge(url, key_url)
          counter += 1
          if counter >= max_number_of_edges:
             break
   # Lets attached each node the size of the in degree that it has
   max_size = max([len(_reverse_mapper.get(node, [])) for node in graph.nodes()])
   # lets make sizes to be between 1 and 10, and lets make it logarithmic in proportion to the graph.in_degree
   # lets make all the labels to be without the prefix of: https://avatar.fandom.com/wiki
   graph = nx.relabel_nodes(graph, lambda x: x.replace('https://avatar.fandom.com/wiki/', ''))

   sizes = [1 + 200 * (len(_reverse_mapper.get(node, [])) / max_size) for node in graph.nodes()]
   return graph, sizes

def draw_graph(graph, sizes):
    """
    This function is responsible for drawing the graph
    :param graph: The graph that we have
    :param sizes: The sizes of the nodes
    :return:
    """
    print('creating positions')
    # lets use the circle layout for the graph
    pos = nx.fruchterman_reingold_layout(graph)
    print('drawing the nodes')
    nx.draw_networkx_nodes(graph, pos, node_size=sizes)
    print('drawing the edges')
    # draw the edges with low alpha to make them lighter, and with smaller width, to make them less visible
    nx.draw_networkx_edges(graph, pos, alpha=1, width=0.2)
    # Lets make the nodes spreader apart by making the figure size bigger and the distance between the nodes bigger
    # let add labels to the nodes
    print('drawing the labels')
    nx.draw_networkx_labels(graph, pos, font_size=0.5)
    print('showing the graph')
    plt.show()
    print('done')

def save_graph_for_gephi(graph):
    """
    This function is responsible for saving the graph for gephi to use
    :param graph: The graph that we have
    :param sizes: The sizes of the nodes
    :return:
    """
    nx.write_graphml(graph, "graph_data.graphml")

def plot_pca(projection: int = 3, number_of_words_to_plot: int = -1,
             connection_to_set_of_words: list = None, focus_on_set: bool = False,
             similarity_list_size: int = 10, plot_with_arrows: bool = True):
    """
    This function is responsible for plotting the PCA of the word vectors
    :return:
    """
    if projection not in [2, 3]:
        raise ValueError("Projection should be either 2 or 3")
    model = getting_model()
    # Get word vectors and corresponding word list
    word_vectors = model.wv.vectors
    words = list(model.wv.key_to_index.keys())

    # Initialize PCA to reduce to 3 dimensions
    pca = PCA(n_components=projection)
    result = pca.fit_transform(word_vectors)

    if number_of_words_to_plot != -1:
        result = result[:number_of_words_to_plot]
        words = words[:number_of_words_to_plot]

    if projection == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # lets do the same lines for the words that are connected just in 2D
        overall_set = set()
        for i, word in enumerate(words):
            if connection_to_set_of_words and word not in connection_to_set_of_words:
                continue
            overall_set.add(word)
            similarity_list = []
            for j, word2 in enumerate(words):
                if i != j:
                    similarity_list.append((model.wv.similarity(word, word2), word2))
            similarity_list.sort(reverse=True)
            for similarity, word2 in similarity_list[:similarity_list_size]:
                if word2 in overall_set:
                    continue
                if plot_with_arrows:
                    ax.plot([result[i, 0], result[words.index(word2), 0]], [result[i, 1], result[words.index(word2), 1]], 'r-', lw=0.1)
                overall_set.add(word2)
        # Create a scatter plot of the projection

        for i, word in enumerate(words):
            if overall_set and word not in overall_set:
                continue
            ax.text(result[i, 0], result[i, 1], word)
            if focus_on_set:
                ax.scatter(result[i, 0], result[i, 1])

        if not focus_on_set:
            ax.scatter(result[:, 0], result[:, 1])

        plt.xlabel('PC1')
        plt.ylabel('PC2')
    else:
        # Create a 3D scatter plot of the projection
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Add annotations for each point
        overall_set = set()
        for i, word in enumerate(words):
            if connection_to_set_of_words and word not in connection_to_set_of_words:
                continue
            overall_set.add(word)
            similarity_list = []
            for j, word2 in enumerate(words):
                if i != j:
                    similarity_list.append((model.wv.similarity(word, word2), word2))
            similarity_list.sort(reverse=True)
            closest_words = [word2 for _, word2 in similarity_list[:similarity_list_size]]
            # lets add those words to the overall set
            overall_set.update(closest_words)
            for closest_word in closest_words:
                closest_word_index = words.index(closest_word)
                if plot_with_arrows:
                    similar_val = model.wv.similarity(word, closest_word)
                    if similar_val < 0.1:
                        continue
                    ax.plot([result[i, 0], result[closest_word_index, 0]],
                            [result[i, 1], result[closest_word_index, 1]],
                            [result[i, 2], result[closest_word_index, 2]], 'purple',
                            alpha=math.sqrt(similar_val),
                            linewidth=math.sqrt(similar_val) * 2)

        for i, word in enumerate(words):
            if overall_set and word not in overall_set:
                continue
            # Lets make sure the text is above the point and above the lines
            ax.text(result[i, 0], result[i, 1], result[i, 2], word, zorder=1, fontsize=8)


            if focus_on_set:
                ax.scatter(result[i, 0], result[i, 1], result[i, 2])

        if not focus_on_set:
            ax.scatter(result[:, 0], result[:, 1], result[:, 2])

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    # Show plot
    plt.show()

def plot_k_means_elbow(words: list, model):
    word_vectors = np.array([model.wv[word] for word in words])
    # Apply t-SNE with 3 components for 3D visualization
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    word_vecs_tsne = tsne.fit_transform(word_vectors)
    wss = []
    k_range = range(1, 50)  # Adjust the range based on your specific dataset
    for k in k_range:
        print(f'Fitting model for k={k}')
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(word_vecs_tsne)
        wss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WSS)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

def plot_k_means_clusters(kmeans: KMeans, model, cluster_numbers_to_plot: list = None,
                          number_of_words_to_plot: int = 10,
                          plot_sphere: bool = False):
    word_clusters, clusters_to_words = get_word_clusters(kmeans, model)

    # lets plot from each cluster 10 words in the same color
    # Lets plot it on a 3d plot
    # lets create a fig in size 10, 10
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    # ax.set_zlim(-10, 10)
    for cluster, words in clusters_to_words.items():
        if cluster_numbers_to_plot and cluster not in cluster_numbers_to_plot:
            continue
        word_vectors = np.array([model.wv[word] for word in words])
        if len(words) < number_of_words_to_plot:
            ax.scatter(word_vectors[:, 0],
                       word_vectors[:, 1],
                       word_vectors[:, 2])
        else:
            ax.scatter(word_vectors[number_of_words_to_plot:, 0],
                       word_vectors[number_of_words_to_plot:, 1],
                       word_vectors[number_of_words_to_plot:, 2])
    # Lets make sure to keep the plot size static so we can see the cluster

    if not plot_sphere:
        plt.legend()
        plt.show()
        return
    # Lets plot a sphere and the centorid of each cluster
    # The sphere should be visable and more like a gas cloud
    # The centroid should be the center of the cluster
    for cluster, words in clusters_to_words.items():
        if cluster_numbers_to_plot and cluster not in cluster_numbers_to_plot:
            continue
        word_vectors = np.array([model.wv[word] for word in words])
        centroid = word_vectors.mean(axis=0)
        most_similar_word = model.wv.most_similar(positive=[centroid], topn=1)
        print(f'Most similar word to centroid of cluster {cluster} is {most_similar_word}')
        ax.text(centroid[0], centroid[1], centroid[2], most_similar_word[0][0], zorder=1, fontsize=8)
        ax.scatter(centroid[0], centroid[1], centroid[2], c='red', s=100, marker='x')
        # Lets plot the sphere
        radius = 0
        for word_vector in word_vectors:
            radius += np.linalg.norm(word_vector - centroid)
        radius /= len(word_vectors)
        radius /= 5
        print(f'Radius for cluster {cluster} is {radius}')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + centroid[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + centroid[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid[2]
        ax.plot_surface(x, y, z, color='b', alpha=0.1)
    plt.legend()
    plt.show()


def plot_number_of_words_per_k_mean(kmeans: KMeans):
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Cluster')
    plt.ylabel('Number of words')
    plt.title('Number of words per cluster')
    plt.show()

def get_word_clusters(kmeans: KMeans, model):
    word_clusters = {word: kmeans.labels_[i] for i, word in enumerate(model.wv.key_to_index.keys())}
    clusters_to_words = defaultdict(list)
    for word, cluster in word_clusters.items():
        clusters_to_words[cluster].append(word)
    return word_clusters, clusters_to_words

def plot_word_simillarity(word1, model, number_of_words=10):
    sim_words = model.wv.most_similar(word1, topn=number_of_words)
    words = [word for word, _ in sim_words]
    sim_values = [value for _, value in sim_words]
    plt.bar(words, sim_values)
    plt.xlabel('Word')
    plt.ylabel('Similarity')
    plt.title(f'Similarity to {word1}')
    plt.show()

def find_cluster_centroid_word(cluster_number, model, kmeans):
    word_clusters, clusters_to_words = get_word_clusters(kmeans, model)
    words = clusters_to_words[cluster_number]
    word_vectors = np.array([model.wv[word] for word in words])
    centroid = word_vectors.mean(axis=0)
    most_similar_word = model.wv.most_similar(positive=[centroid], topn=1)
    return most_similar_word[0]

def create_tsne_words_vectors(model):
    word_vectors = np.array([model.wv[word] for word in model.wv.key_to_index.keys()])
    tsne = TSNE(n_components=3, random_state=42)
    word_vectors = tsne.fit_transform(word_vectors)
    return word_vectors

def plot_t_sne_clusters_3d_with_kmeans_and_arrows(model, number_of_words_to_plot: int = 100, n_clusters: int = 5,
                                                  number_of_powerful_words: int = 5, good_speach_parts: list= None):
    # Extract word vectors and corresponding words
    words = list(model.wv.key_to_index.keys())

    if good_speach_parts:
        print('Filtering words by speach parts')
        nlp = spacy.load('en_core_web_sm')
        whole_words = nlp(' '.join(words))

        speach_parts_to_words = defaultdict(str)
        for word in whole_words:
            speach_parts_to_words[word.text] = word.pos_

        words = [word for word in words if speach_parts_to_words[word] in good_speach_parts]
        print('done filtering')
    word_vectors = np.array([model.wv[word] for word in words])

    # Apply t-SNE with 3 components for 3D visualization
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    word_vecs_tsne = tsne.fit_transform(word_vectors)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(word_vecs_tsne)

    # Plotting
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n_clusters))

    # Plot each point and color it according to its cluster label
    for i, word in enumerate(words):
        if i < number_of_words_to_plot:  # Limit the number of words to plot
            ax.scatter(word_vecs_tsne[i, 0], word_vecs_tsne[i, 1], word_vecs_tsne[i, 2], color=colors[labels[i]], alpha=0.5)

    # Calculate and plot centroids and arrows
    # Inside your loop over each cluster:
    total_words = list(model.wv.key_to_index.keys())
    powerful_words = defaultdict(list)
    for cluster in range(n_clusters):
        cluster_points = word_vecs_tsne[labels == cluster]
        centroid = np.mean(cluster_points, axis=0)
        # Lets find the closest word to the centroid from the word_vecs_tsne
        cluster_words = [words[i] for i in range(len(words)) if labels[i] == cluster]

        # lets save from cluster the most powerful words of the model
        counter = number_of_powerful_words
        while counter > 0:
            break_flag = False
            for word in total_words:
                if word in cluster_words:
                    powerful_words[cluster].append(word)
                    total_words.remove(word)
                    counter -= 1
                    break_flag = True
                    break
            # If we are here, it means that we have not found any word in the cluster so we need to break the loop
            if not break_flag:
                break

        ax.scatter(centroid[0], centroid[1], centroid[2], color=colors[cluster], s=500, marker='o', edgecolors='k',
                   linewidths=2)
        ax.text(centroid[0], centroid[1], centroid[2], f'{powerful_words[cluster][0]}',
                color='k', fontsize=14, fontweight='bold',
                zorder=1000, bbox=dict(facecolor='white', alpha=0.5))

        # Plotting arrows
        for point in cluster_points:
            ax.quiver(centroid[0], centroid[1], centroid[2], point[0] - centroid[0], point[1] - centroid[1],
                      point[2] - centroid[2], color=colors[cluster], alpha=0.3, linewidth=0.5, arrow_length_ratio=0.01)
    # lets add the most powerful words in the side of the plot

    ax.legend(powerful_words.items(), title='Most powerful words', loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('3D t-SNE visualization with K-Means Clustering and Centroid Arrows')
    plt.show()

def get_only_part_of_speach_words(model, speach_part: list):
    nlp = spacy.load('en_core_web_sm')
    whole_words = nlp(' '.join(list(model.wv.key_to_index.keys())))
    words_to_return = []
    for word in whole_words:
        if word.pos_ in speach_part:
            words_to_return.append(word.text)
    return words_to_return
