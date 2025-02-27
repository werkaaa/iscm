from results_reproducibility.definitions import *


def map_data_gen_method_names(data_gen_method, short=False):
    if data_gen_method in ['no', 'raw']:
        return 'SCM'
    elif data_gen_method in ['alternate']:
        return 'iSCM'
    elif data_gen_method in ['final']:
        if short:
            return 'Stand. SCM'
        else:
            return 'Standardized SCM'
    else:
        raise NotImplementedError(f'Data gen method {data_gen_method} not known')

def map_heuristic_data_gen_method_names(data_gen_method, short=False):
    if data_gen_method in ['mooij']:
        return 'Mooij'
    elif data_gen_method in ['mooij_noise']:
        return 'Mooij + noise scaling'
    elif data_gen_method in ['squires']:
        return 'Squires'
    else:
        raise NotImplementedError(f'Data gen method {data_gen_method} not known')



def map_method_names(method):
    if method in [NOTEARS]:
        return '$\\textsc{Notears}$'
    elif method in [AVICI]:
        return '$\\textsc{Avici}$'
    elif method in [VAR_SORT_REGRESS]:
        return 'Var-\\textsc{sr}'
    elif method in [R2_SORT_REGRESS]:
        return r'$\operatorname{R}^2$-$\textsc{sr}$'
    elif method in [RANDOM_SORT_REGRESS]:
        return 'Rand.-$\\textsc{sr}$'
    elif method in [PC]:
        return '$\\textsc{Pc}$'
    elif method in [GES]:
        return '$\\textsc{Ges}$'
    elif method in [GOLEM_NV]:
        return '$\\textsc{Golem}$ \n $\\textsc{nv}$'
    elif method in [GOLEM_EV]:
        return '$\\textsc{Golem}$ \n $\\textsc{ev}$'
    elif method in [LINGAM]:
        return '$\\textsc{Lingam}$'
    elif method in [SCORE]:
        return '$\\textsc{Score}$'
    elif method in [CAM]:
        return '$\\textsc{Cam}$'
    else:
        raise NotImplementedError(f'Causal learning method {method} not known')

def map_sortability_name(method):
    if method in [R2_SORTABILITY]:
        return '$\operatorname{R}^2$-sortability'
    if method in [VAR_SORTABILITY]:
        return '$\operatorname{Var}$-sortability'
    elif method in [CEV_SORTABILITY]:
        return '$\operatorname{CEV_f}$-sortability'
    elif method in [INDG_SORTABILITY]:
        return 'ID-sortability'
    elif method in [NS_SORTABILITY]:
        return 'NS-sortability'
    elif method in [MBS_SORTABILITY]:
        return 'MBS-sortability'
    else:
        raise NotImplementedError(f'Causal learning method {method} not known')


def get_filename_from_data_meta(meta_info):
    file_name = '_'.join([f'{v}' for k, v in meta_info.items() if k != 'data_folder'])
    file_name = file_name.replace(' ', '_').replace(',', '').replace('[', '').replace(']', '')
    return file_name


def format_metric(metric):
    if metric in ['f1', 'precision', 'recall']:
        return metric.capitalize()
    elif metric in ['shd']:
        return 'SHD'
    else:
        raise NotImplementedError(f'Metric {metric} not known.')

def format_weight_range(weight_range):
    if isinstance(weight_range, str):
        return f'$\pm {weight_range}$'
    return f'$\pm [{weight_range[0]},{weight_range[1]}]$'
