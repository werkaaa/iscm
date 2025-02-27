from sklearn.metrics import precision_score, recall_score, f1_score


def precision(true_graph, predicted_graph):
    assert true_graph.ndim == predicted_graph.ndim
    true_graph = true_graph.astype(float)
    predicted_graph = predicted_graph.astype(float)
    return precision_score(true_graph.ravel(), predicted_graph.ravel())


def recall(true_graph, predicted_graph):
    assert true_graph.ndim == predicted_graph.ndim
    true_graph = true_graph.astype(float)
    predicted_graph = predicted_graph.astype(float)
    return recall_score(true_graph.ravel(), predicted_graph.ravel())


def f1(true_graph, predicted_graph):
    assert true_graph.ndim == predicted_graph.ndim
    true_graph = true_graph.astype(float)
    predicted_graph = predicted_graph.astype(float)
    return f1_score(true_graph.ravel(), predicted_graph.ravel())
