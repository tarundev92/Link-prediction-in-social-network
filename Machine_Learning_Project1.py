import networkx as nx
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.externals import joblib
from sklearn import metrics
import time
import random
import csv
import math
import pickle
from multiprocessing import Process, Queue


def set_random_numbers_for_samples(max_value, max_value2):
    # random_positive_samples = random.sample(range(1, max_value), 30000)
    # random1 = random.sample(range(1, max_value), 5)
    # random2 = random.sample(range(1, 100), 10)
    # return random1, random2

    random_positive_samples = random.sample(range(1, max_value), 100000)
    random_negative_samples = random.sample(range(1, max_value2), 100000)
    # random_negative_samples2 = random.sample(range(1, max_value2), 30000)
    return random_positive_samples, random_negative_samples, 0


def non_neighbors(graph, node):
    """Returns the non-neighbors of the node in the graph.

    Parameters
    ----------
    graph : NetworkX graph
        Graph to find neighbors.

    node : node
        The node whose neighbors will be returned.

    Returns
    -------
    non_neighbors : iterator
        Iterator of nodes in the graph that are not neighbors of the node.
    """
    nbors = set(graph.neighbors(node)) | set([node])
    return (nnode for nnode in graph if nnode not in nbors)


def non_edges(DirectedGraph):
    non_edges_list = []
    non_edges_list_no_path = []
    non_edges_list_limit = 2394660
    temp_count = 0
    for u in DirectedGraph.nodes():
        in_count = 0
        for v in non_neighbors(DirectedGraph, u):
            non_edges_list.append((u, v))
            if temp_count == non_edges_list_limit:
                break
            in_count += 1
            temp_count += 1
            if in_count == 100:
                break
        if temp_count == non_edges_list_limit:
            break
    print("non_edges_list_no_path:", len(non_edges_list_no_path))
    # return non_edges_list_no_path, non_edges_list
    return non_edges_list, non_edges_list_no_path


def generate_feature_values(DirectedGraph, samples, dataset_queue, is_positive=True, is_id=True, start_id=1, is_test_set=False):
    print("generate_feature_values parallel execution")
    decimal_round = 7
    type_sample = 1 if is_positive else 0
    data_set = []
    temp_i = 1
    for sample in samples:
        Source = sample[0]
        Sink = sample[1]

        data = {}

        if is_id:
            data["Id"] = start_id
            start_id += 1

        data["Source"] = Source
        data["Sink"] = Sink

        Source_followers_edges = list(DirectedGraph.in_edges(Source))
        Sink_followers_edges = list(DirectedGraph.in_edges(Sink))

        Source_followers_nodes = set([edge[0] for edge in Source_followers_edges])
        Sink_followers_nodes = set([edge[0] for edge in Sink_followers_edges])

        data["Source_followers"] = len(Source_followers_edges)
        data["Sink_followers"] = len(Sink_followers_edges)

        Source_following_edges = list(DirectedGraph.out_edges(Source))
        Sink_following_edges = list(DirectedGraph.out_edges(Sink))

        Source_following_nodes = set([edge[1] for edge in Source_following_edges])
        Sink_following_nodes = set([edge[1] for edge in Sink_following_edges])


        data["Source_following"] = len(Source_following_nodes)
        data["Sink_following"] = len(Sink_following_nodes)

        Source_degree = len(Source_followers_edges + Source_following_edges)
        Sink_degree = len(Sink_followers_edges + Sink_following_edges)

        Source_neigh = sorted(nx.all_neighbors(DirectedGraph, Source))
        data["Source_total_neigh"] = len(Source_neigh)
        Sink_neigh = sorted(nx.all_neighbors(DirectedGraph, Sink))
        data["Sink_total_neigh"] = len(Sink_neigh)

        common_neigh = set(Source_neigh).intersection(set(Sink_neigh))

        data["Source_neigh_density"] = (1.0 / math.log(data["Source_total_neigh"]+1)) if data["Source_total_neigh"] != 0 else 0
        data["Sink_neigh_density"] = (1.0 / math.log(data["Sink_total_neigh"]+1)) if data["Sink_total_neigh"] != 0 else 0

        data["Salton"] = 0.0 + len(common_neigh) / math.sqrt(data["Source_total_neigh"] * data["Sink_total_neigh"]) if len(common_neigh) != 0 else 0


        if len(common_neigh) == 0:
            data["Cosine_similarity"] = 0
        else:
            data["Cosine_similarity"] = 0.0 + len(common_neigh) / (len(Source_neigh) * len(Sink_neigh))

        if len(common_neigh) == 0:
            data["Sorsen_index"] = 0
        else:
            data["Sorsen_index"] = 0.0 + (2 * len(common_neigh)) / (len(Source_neigh) + len(Sink_neigh))

        if len(common_neigh) == 0:
            data["Hub_promoted_index"] = 0
        else:
            data["Hub_promoted_index"] = 0.0 + len(common_neigh) / min(len(Source_neigh), len(Sink_neigh))

        if len(common_neigh) == 0:
            data["Hub_depressed_index"] = 0
        else:
            data["Hub_depressed_index"] = 0.0 + len(common_neigh) / max(len(Source_neigh), len(Sink_neigh))

        if len(common_neigh) == 0:
            data["Leicht_Holme_Newman_Index"] = 0
        else:
            data["Leicht_Holme_Newman_Index"] = 0.0 + len(common_neigh) / (len(Source_neigh) * len(Sink_neigh))

        data["Source_followers_density"] = round(data["Source_followers"] / Source_degree, decimal_round)
        data["Sink_followers_density"] = round(data["Sink_followers"] / Sink_degree, decimal_round)


        data["Source_following_density"] = round(data["Source_following"] / Source_degree, decimal_round)
        data["Sink_following_density"] = round(data["Sink_following"] / Sink_degree, decimal_round)

        Source_bi_degree = Source_followers_nodes.intersection(Source_following_nodes)
        data["Source_bi_degree_density"] = round((len(Source_bi_degree) / 2) / Source_degree, decimal_round)

        Sink_bi_degree = Sink_followers_nodes.intersection(Sink_following_nodes)
        data["Sink_bi_degree_density"] = round((len(Sink_bi_degree) / 2) / Sink_degree, decimal_round)

        Source_bi_list = []
        for Source_follower in Source_followers_nodes:
            if DirectedGraph.has_edge(Source, Source_follower):
                Source_bi_list.append(Source_follower)

        data["Source_bi"] = len(Source_bi_list)

        Sink_bi_list = []
        for Sink_follower in Sink_followers_nodes:
            if DirectedGraph.has_edge(Source, Sink_follower):
                Sink_bi_list.append(Sink_follower)

        data["Sink_bi"] = len(Sink_bi_list)
        data["Common_followers"] = len(Source_followers_nodes.intersection(Sink_followers_nodes))
        data["Common_following"] = len(Source_following_nodes.intersection(Sink_following_nodes))
        data["Common_bi"] = len(set(Source_bi_list).intersection(set(Sink_bi_list)))


        try:
            data["Shortest_path"] = len(nx.shortest_path(DirectedGraph, Source, Sink)) - 1 if is_positive else 0
        except nx.NetworkXNoPath:
            data["Shortest_path"] = 0



        data["Total_followers"] = data["Source_followers"] + data["Sink_followers"]
        data["Total_following"] = data["Source_following"] + data["Sink_following"]

        # print("fea total done")

        data["Friend_measure"] = 0

        for Source_neighbour in list(Source_followers_nodes | Source_following_nodes):
            for Sink_neighbour in list(Sink_followers_nodes | Sink_following_nodes):
                if DirectedGraph.has_edge(Source_neighbour, Sink_neighbour) or DirectedGraph.has_edge(Sink_neighbour, Source_neighbour):
                    data["Friend_measure"] += 1

        try:
            Source_nodes = sorted(nx.all_neighbors(DirectedGraph, Source))
            Sink_nodes = sorted(nx.all_neighbors(DirectedGraph, Sink))
            data["Jaccard_coefficient"] = round(len(set(Source_nodes).intersection(set(Sink_nodes))) / len(set(Source_nodes) | set(Sink_nodes)), decimal_round)
        except ZeroDivisionError:
            data["Jaccard_coefficient"] = 0

        # print("fea jaccard done")

        data["Preferential_attachment_score"] = round(data["Source_followers"] * data["Sink_followers"], decimal_round)


        data["Adamic_adar"] = 0
        data["Resource_allocation"] = 0
        for node in (Source_followers_nodes | Source_following_nodes).intersection(Sink_followers_nodes | Sink_following_nodes):
            try:
                data["Adamic_adar"] += round(1.0/math.log(DirectedGraph.degree(node)), decimal_round)
            except ZeroDivisionError:
                pass

            try:
                data["Resource_allocation"] += round(1/DirectedGraph.degree(node), decimal_round)
            except ZeroDivisionError:
                pass

        data["Transitive_friends"] = round(len(Source_following_nodes.intersection(Sink_followers_nodes)), decimal_round)

        data["Opposite"] = 1 if DirectedGraph.has_edge(Sink, Source) else 0


        if not is_test_set:
            data["type_sample"] = type_sample

        data_set.append(data)
        if temp_i % 1000 == 0:
            temp_i = 0
            print("1000 set done")

        temp_i += 1

    dataset_queue.put(data_set)
    # return data_set


def generate_csv_from_dict(file_name, fieldnames, dataset):
    with open(file_name + '.csv', 'w', newline='') as csvfile:
        # fieldnames = ['first_name', 'last_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)



def construct_graph_save_to_txt():
    graph = open('train.txt', 'r').readlines()
    DirectedGraph = nx.DiGraph()
    print("Constructing graph...")
    for g in graph:
        sub_graph = g.strip().split("	")
        node = int(sub_graph.pop(0))
        edges = [(node, int(node2)) for node2 in sub_graph]
        DirectedGraph.add_node(node)
        DirectedGraph.add_edges_from(edges)
    nx.write_gpickle(DirectedGraph, "Graph.gpickle")
    print("Graph construction done.")


def randomly_select_positive_negative_edges():
    start_time = time.time()
    DirectedGraph = nx.read_gpickle("Graph.gpickle")
    print("Graph loaded from file. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    all_edges = list(nx.edges(DirectedGraph))

    non_edges_list, non_edges_list_spath_2 = non_edges(DirectedGraph)

    random_positive_samples, random_negative_samples, random_negative_samples2 = set_random_numbers_for_samples(
        len(all_edges), len(non_edges_list))


    positive_samples = []
    negative_samples = []
    print("Randomly selecting positive and negative samples for train set")

    for i, j in zip(random_positive_samples, random_negative_samples):
        positive_samples.append({"Source": all_edges[i][0], "Sink": all_edges[i][1]})
        negative_samples.append({"Source": non_edges_list[j][0], "Sink": non_edges_list[j][1]})

    generate_csv_from_dict(file_name="train_positive_samples", fieldnames=positive_samples[0].keys(), dataset=positive_samples)
    generate_csv_from_dict(file_name="train_negative_samples", fieldnames=negative_samples[0].keys(), dataset=negative_samples)
    print("Done. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))



def extract_features():
    start_time = time.time()
    raw_test = open('test-public.txt', 'r').readlines()
    raw_test.pop(0)
    test_samples = [(int(line.strip().split("	")[1]), int(line.strip().split("	")[2])) for line in raw_test]
    i = 0
    is_id = False
    print("Loading graph from file start_time:", time.strftime("%H:%M:%S", time.gmtime(start_time)))
    DirectedGraph = nx.read_gpickle("Graph.gpickle")
    print("Graph loaded from file. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    positive_samples = [tuple(tup) for tup in pd.read_csv('train_positive_samples.csv').values.tolist()]
    negative_samples = [tuple(tup) for tup in pd.read_csv('train_negative_samples.csv').values.tolist()]
    negative_samples2 = []

    positive_samples = list(set(positive_samples))
    negative_samples = list(set(negative_samples))

    DirectedGraph.remove_edges_from(positive_samples)


    print("Generating features for train set, test set parallelly. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    positive_batch_size = int(len(positive_samples) / 4)
    negative_batch_size = int(len(negative_samples) / 4)
    negative2_batch_size = int(len(negative_samples2) / 4)

    dataset_queue = Queue()
    train_positive_batch0 = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': positive_samples[:positive_batch_size - 1], 'is_id': is_id, 'dataset_queue': dataset_queue})
    train_positive_batch1 = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': positive_samples[positive_batch_size:(positive_batch_size*2) - 1], 'is_id': is_id, 'dataset_queue': dataset_queue})
    train_positive_batch2 = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': positive_samples[(positive_batch_size*2):(positive_batch_size*3) - 1], 'is_id': is_id, 'dataset_queue': dataset_queue})
    train_positive_batch3 = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': positive_samples[(positive_batch_size*3):(positive_batch_size*4) - 1], 'is_id': is_id, 'dataset_queue': dataset_queue})
    train_negative_batch0 = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': negative_samples[:negative_batch_size - 1], 'is_id': is_id, 'is_positive': False, 'start_id': len(positive_samples) + 1, 'dataset_queue': dataset_queue})
    train_negative_batch1 = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': negative_samples[negative_batch_size:(negative_batch_size*2) - 1], 'is_id': is_id, 'is_positive': False, 'start_id': len(positive_samples) + 1, 'dataset_queue': dataset_queue})
    train_negative_batch2 = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': negative_samples[(negative_batch_size*2):(negative_batch_size*3) - 1], 'is_id': is_id, 'is_positive': False, 'start_id': len(positive_samples) + 1, 'dataset_queue': dataset_queue})
    train_negative_batch3 = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': negative_samples[(negative_batch_size*3):(negative_batch_size*4) - 1], 'is_id': is_id, 'is_positive': False, 'start_id': len(positive_samples) + 1, 'dataset_queue': dataset_queue})


    train_positive_batch0.start()
    train_positive_batch1.start()
    train_positive_batch2.start()
    train_positive_batch3.start()
    train_negative_batch0.start()
    train_negative_batch1.start()
    train_negative_batch2.start()
    train_negative_batch3.start()


    train_data_set = dataset_queue.get() + dataset_queue.get() + dataset_queue.get() + dataset_queue.get() + \
                     dataset_queue.get() + dataset_queue.get() + dataset_queue.get() + dataset_queue.get()

    print("Train set generation done. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    test_batch = Process(target=generate_feature_values, kwargs={'DirectedGraph': DirectedGraph, 'samples': test_samples, 'is_id': is_id, 'is_test_set': True, 'is_positive': False, 'dataset_queue': dataset_queue})
    test_batch.start()
    test_data_set = dataset_queue.get()

    field_names = list(test_data_set[0].keys())

    print("Done. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    if is_id:
        field_names.insert(0, "Id")

    print("Generating train CSV")
    train_field_names = field_names + ["type_sample"]
    generate_csv_from_dict(dataset=train_data_set, fieldnames=train_field_names, file_name="train_dataset")
    print("Done.")
    print("Generating test CSV")
    generate_csv_from_dict(dataset=test_data_set, fieldnames=field_names, file_name="test_dataset")
    print("Done.")

    print("Total time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return 0


def generate_result_csv_from_prediction(prediction, file_name="test_result"):
    field_names = ["Id", "Prediction"]
    test_result = []
    i = 1
    for x in np.nditer(prediction):
        test_result.append({"Id": i, "Prediction": x})
        i += 1
    generate_csv_from_dict(dataset=test_result, fieldnames=field_names, file_name=file_name)



def get_all_classifier_result():
    # Creating Dataset and including the first row by setting no header as input
    train_dataset = pd.read_csv('train_dataset.csv')
    test_dataset = pd.read_csv('test_dataset.csv')

    # Splitting the data into independent and dependent variables
    X = train_dataset.iloc[:, 2:len(train_dataset.columns) - 1].values
    y = train_dataset.iloc[:, len(train_dataset.columns) - 1].values
    # print("-----------------------Chk columns:", X.columns)

    # Creating the Training and Test set from data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
    X_train = X
    y_train = y
    X_test = test_dataset.iloc[:, 2:len(test_dataset.columns)].values
    X_te = X_test

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    random_forest_classifier(X_train=X_train, y_train=y_train, X_test=X_test, test_dataset=test_dataset, validation=False)
    random_forest_regressor(X, y, X_te)
    bagging_classifier(X_train=X_train, y_train=y_train, X_test=X_test, test_dataset=test_dataset, validation=False)
    adaboost_classifier(X_train=X_train, y_train=y_train, X_test=X_test, test_dataset=test_dataset, validation=False)
    gradientboost_classifier(X_train, y_train, X_test)


def validation():
    train_dataset = pd.read_csv('train_dataset.csv')
    test_dataset = pd.read_csv('test_dataset.csv')

    X = train_dataset.iloc[:, 2:len(train_dataset.columns) - 1].values
    y = train_dataset.iloc[:, len(train_dataset.columns) - 1].values

    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

    for train_index, test_index in kf.split(X):
        # print("Train:", train_index, "Validation:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    random_forest_classifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, test_dataset=test_dataset,
                             validation=True)
    bagging_classifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, test_dataset=test_dataset,
                       validation=True)
    adaboost_classifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, test_dataset=test_dataset, validation=True)


def random_forest_classifier(X_train, y_train, X_test, y_test, test_dataset, validation):
    print("---------------------RandomForestClassifier--------------------------")
    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)

    feature_importances = pd.DataFrame(classifier.feature_importances_,
                                       index=test_dataset.columns[2:],
                                       columns=['importance']).sort_values('importance', ascending=False)

    print("Feature Importance:", feature_importances)
    # Predicting the Test set results
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = classifier.predict(X_test)

    if not validation:
        generate_result_csv_from_prediction(y_pred_proba, file_name="test_result_random_forest")

    if validation:
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        print("accuracy_score:", metrics.accuracy_score(y_test, y_pred))
        print("roc_auc_score:", roc_auc_score(y_test, y_pred_proba))

    print("---------------------RandomForestClassifier Done--------------------------")


def bagging_classifier(X_train, y_train, X_test, y_test, test_dataset, validation):
    print("---------------------BaggingClassifier--------------------------")
    # Fitting Random Forest Classification to the Training set
    classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, oob_score=True)
    classifier.fit(X_train, y_train)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = classifier.predict(X_test)

    if not validation:
        generate_result_csv_from_prediction(y_pred_proba, file_name="test_result_bagging")

    if validation:
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        print("accuracy_score:", metrics.accuracy_score(y_test, y_pred))
        print("roc_auc_score:", roc_auc_score(y_test, y_pred_proba))
    print("---------------------BaggingClassifier Done--------------------------")


construct_graph_save_to_txt()
randomly_select_positive_negative_edges()
extract_features()
get_all_classifier_result()
validation()

