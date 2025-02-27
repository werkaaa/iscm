import pandas as pd
from CausalDisco.baselines import r2_sort_regress, var_sort_regress, random_sort_regress
from CausalDisco.analytics import r2_sortability, snr_sortability, order_alignment, var_sortability

try:
    import avici
except Exception:
    pass

try:
    from graph_discovery_algo.cdt import run_pc as run_pc_from_cdt
    from graph_discovery_algo.cdt import run_ges as run_ges_from_cdt
    from graph_discovery_algo.cdt import run_lingam as run_lingam_from_cdt
except Exception:
    pass

try:
    from graph_discovery_algo.golem.src.golem import golem
except Exception:
    pass

try:
    from dodiscover import make_context
    from dodiscover.toporder.score import SCORE
    from dodiscover.toporder.cam import CAM
    import networkx as nx
except Exception:
    pass

from graph_discovery_algo.notears import notears_linear

from data_generation.graph_utils import convert_to_binary_matrix, get_in_degrees, get_neighbour_counts, \
    get_markov_blanket_sizes
from results_reproducibility.definitions import PROJECT_DIR, AVICI_CHECKPOINT_LINEAR, AVICI_CHECKPOINT_RFF, LINEAR, RFF


def run_notears(data, spec):
    args = {
        'loss_type': 'l2',
        'max_iter': 100,
        'h_tol': 1e-8,
        'rho_max': 1e+16,
        'w_threshold': 0.3
    }
    for key, value in spec.items():
        if key.startswith('__'):
            continue
        if isinstance(value, dict):
            m = data["meta_info"]
            if 'weight_range' in m.keys():
                data_key = f"{m['data_gen_method']}_w_{m['weight_range'][0]}_{m['weight_range'][1]}_n_{m['noise_variance_range'][0]}_{m['noise_variance_range'][0]}"
            else:
                data_key = m['data_gen_method']
            args[key] = value[data_key]
        else:
            args[key] = value
    p = {'predictions': notears_linear(data["sample"], **args)}
    return p


def run_avici(data, spec):
    if spec["model"] == LINEAR:
        model = avici.load_pretrained(
            checkpoint_dir=PROJECT_DIR / AVICI_CHECKPOINT_LINEAR,
            expects_counts=False)
    elif spec["model"] == RFF:
        model = avici.load_pretrained(
            checkpoint_dir=PROJECT_DIR / AVICI_CHECKPOINT_RFF,
            expects_counts=False)
    else:
        raise NotImplementedError(f"AVICI model {spec['model']} not known.")

    if "num_samples" in spec.keys():
        sample = data["sample"][:spec["num_samples"]]
    else:
        sample = data["sample"]
    prob_est = model(sample)
    p = {'predictions': convert_to_binary_matrix(prob_est > spec["prob_threshold"])}
    return p


def run_pc(data, spec, seed):
    args = {}
    for key, value in spec.items():
        if key.startswith('__'):
            continue
        if isinstance(value, dict):
            m = data["meta_info"]
            if 'weight_range' in m.keys():
                data_key = f"{m['data_gen_method']}_w_{m['weight_range'][0]}_{m['weight_range'][1]}_n_{m['noise_variance_range'][0]}_{m['noise_variance_range'][0]}"
            else:
                data_key = m['data_gen_method']
            args[key] = value[data_key]
        else:
            args[key] = value

    predictions = run_pc_from_cdt(seed, data["sample"], args)
    print(predictions, type(predictions))
    return {'predictions': predictions}


def run_ges(data, spec, seed):
    predictions = run_ges_from_cdt(seed, data["sample"], spec)
    print(predictions, type(predictions))
    return {'predictions': predictions}

def run_lingam(data, spec, seed):
    predictions = run_lingam_from_cdt(seed, data["sample"], spec)
    print(predictions, type(predictions))
    return {'predictions': predictions}

def run_score(data, spec):
    data_dataframe = pd.DataFrame(data["sample"])
    print(list(data_dataframe.columns), 'columns')
    context = make_context().variables(data=data_dataframe).build()
    print(context.get_augmented_nodes(), 'nodes')
    print(list(spec.keys()), 'keys')
    print(spec['alpha'], 'alpha')
    score = SCORE(alpha=spec['alpha'], pns=True)
    score.learn_graph(data_dataframe, context)
    print(type(score.graph_))
    print(score.graph_)
    print(score.graph_.nodes)
    predictions = nx.adjacency_matrix(score.graph_).toarray()
    print(predictions)
    return {'predictions': predictions}

def run_cam(data, spec):
    data_dataframe = pd.DataFrame(data["sample"])
    context = make_context().variables(data=data_dataframe).build()

    args = {}
    for key, value in spec.items():
        if key.startswith('__'):
            continue
        if isinstance(value, dict):
            m = data["meta_info"]
            if 'weight_range' in m.keys():
                data_key = f"{m['data_gen_method']}_w_{m['weight_range'][0]}_{m['weight_range'][1]}_n_{m['noise_variance_range'][0]}_{m['noise_variance_range'][0]}"
            else:
                data_key = m['data_gen_method']
            args[key] = value[data_key]
        else:
            args[key] = value


    score = CAM(pns=True, alpha=args['alpha'], n_splines=args['nsplines'], splines_degree=args['degsplines'])
    score.learn_graph(data_dataframe, context)
    predictions = nx.adjacency_matrix(score.graph_).toarray()
    print(predictions)
    return {'predictions': predictions}

def run_golem(data, spec, seed):
    from graph_discovery_algo.golem.src.utils.train import postprocess

    args = {}
    for key, value in spec.items():
        if key.startswith('__'):
            continue
        if isinstance(value, dict):
            m = data["meta_info"]
            if 'weight_range' in m.keys():
                data_key = f"{m['data_gen_method']}_w_{m['weight_range'][0]}_{m['weight_range'][1]}_n_{m['noise_variance_range'][0]}_{m['noise_variance_range'][0]}"
            else:
                data_key = m['data_gen_method']
            args[key] = value[data_key]
        else:
            args[key] = value

    B_est = golem(data["sample"], lambda_1=args['lambda_1'], lambda_2=args['lambda_2'], equal_variances=args['equal_variances'],
                  learning_rate=args['learning_rate'], seed=seed)

    # Post-process estimated solution and compute results
    B_processed = postprocess(B_est, graph_thres=args['w_threshold'])
    return {'predictions': B_processed}


def run_var_sort_regress(data, spec):
    p = {'predictions': var_sort_regress(data["sample"])}
    return p


def run_r2_sort_regress(data, spec):
    p = {'predictions': r2_sort_regress(data["sample"])}
    return p


def run_random_sort_regress(data, spec, seed):
    p = {'predictions': random_sort_regress(data["sample"], seed=seed)}
    return p


def run_r2_sortability(data, spec):
    p = {'predictions': r2_sortability(data["sample"], data["g"])}
    return p


def run_var_sortability(data, spec):
    p = {'predictions': var_sortability(data["sample"], data["g"])}
    return p


def run_cev_sortability(data, spec):
    p = {'predictions': snr_sortability(data["sample"], data["g"])}
    return p


def run_indg_sortability(data, spec):
    def in_degree_sortability(W, tol=0.):
        scores = get_in_degrees(W)
        return order_alignment(W, scores, tol=tol)

    p = {'predictions': in_degree_sortability(data["g"])}
    return p


def run_neighbour_count_sortability(data, spec):
    def nc_sortability(W, tol=0.):
        scores = get_neighbour_counts(W)
        return order_alignment(W, scores, tol=tol)

    p = {'predictions': nc_sortability(data["g"])}
    return p


def run_markov_blanket_size_sortability(data, spec):
    def nc_sortability(W, tol=0.):
        scores = get_markov_blanket_sizes(W)
        return order_alignment(W, scores, tol=tol)

    p = {'predictions': nc_sortability(data["g"])}
    return p
