"""Dataset-specific overrides for the shared simulation NRT search."""
from copy import deepcopy
from dataclasses import asdict, fields
import math

from regularization.tuning.main import SearchConfig

DATASETS = ('gt_2011', 'rotterdam_recur', 'rotterdam_death')
SEARCH_FIELDS = {field.name for field in fields(SearchConfig)}
BASELINE_FIELDS = {'depth', 'shared_layers', 'shared_width', 'private_layers',
                   'private_width', 'regularization_strength'}


def _section(value, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be a JSON object')
    unknown = set(value) - {'search', 'baseline'}
    if unknown:
        raise ValueError(f'{where}: unknown sections {sorted(unknown)}')
    for name, allowed in (('search', SEARCH_FIELDS), ('baseline', BASELINE_FIELDS)):
        child = value.get(name, {})
        if not isinstance(child, dict):
            raise ValueError(f'{where}.{name} must be a JSON object')
        if set(child) - allowed:
            raise ValueError(f'{where}.{name}: unknown fields {sorted(set(child)-allowed)}')
    return value


def _integer(value, minimum, where):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{where} must be an integer >= {minimum}')


def _strength(value, where):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{where} must be finite and nonnegative')


def resolve_search_spaces(config, overrides=None):
    """Merge defaults and endpoint overrides, projecting omitted anchors into range."""
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError('NRT search spaces must be a JSON object')
    unknown = set(overrides) - {'defaults', *DATASETS}
    if unknown:
        raise ValueError(f'Unknown NRT datasets: {sorted(unknown)}; use {DATASETS}')
    common = _section(overrides.get('defaults', {}), 'defaults')
    result = {}
    for dataset in DATASETS:
        local = _section(overrides.get(dataset, {}), dataset)
        search = {**config['nrt_search'], **common.get('search', {}), **local.get('search', {})}
        for key in SEARCH_FIELDS - {'widths', 'strength_max'}:
            minimum = 0 if key.endswith('_seed') or key == 'shared_layers_max' else 1
            _integer(search[key], minimum, f'{dataset}.search.{key}')
        for key in ('fold_seed', 'search_seed', 'search_fit_seed'):
            if search[key] >= 2**32:
                raise ValueError(f'{dataset}.search.{key} must be < 2**32')
        widths = search['widths']
        if not isinstance(widths, (list, tuple)) or not widths:
            raise ValueError(f'{dataset}.search.widths must be a nonempty list')
        for width in widths:
            _integer(width, 1, f'{dataset}.search.widths entry')
        if len(set(widths)) != len(widths):
            raise ValueError(f'{dataset}.search.widths must not contain duplicates')
        search['widths'] = tuple(widths)
        _strength(search['strength_max'], f'{dataset}.search.strength_max')
        if (search['depth_min'] > search['depth_max'] or search['folds'] < 2
                or search['initial_trials'] > search['trials']):
            raise ValueError(f'{dataset}: require depth_min <= depth_max, folds >= 2, '
                             'and initial_trials <= trials')

        explicit = {**common.get('baseline', {}), **local.get('baseline', {})}
        baseline = {**config['nrt_baseline'], **explicit}
        intervals = {'depth': (search['depth_min'], search['depth_max']),
                     'shared_layers': (0, search['shared_layers_max']),
                     'private_layers': (1, search['private_layers_max']),
                     'regularization_strength': (0, search['strength_max'])}
        for key, (lower, upper) in intervals.items():
            if key == 'regularization_strength':
                _strength(baseline[key], f'{dataset}.baseline.{key}')
            else:
                _integer(baseline[key], lower if key in explicit else 0, f'{dataset}.baseline.{key}')
            if key not in explicit:
                baseline[key] = min(upper, max(lower, baseline[key]))
            elif not lower <= baseline[key] <= upper:
                raise ValueError(f'{dataset}.baseline.{key} must be within [{lower}, {upper}]')
        for key in ('shared_width', 'private_width'):
            _integer(baseline[key], 1, f'{dataset}.baseline.{key}')
            if baseline[key] not in widths:
                if key in explicit:
                    raise ValueError(f'{dataset}.baseline.{key} must belong to search.widths')
                baseline[key] = min(widths, key=lambda width: (abs(width-baseline[key]), width))
        # x_dim and geometry constraints are supplied by the existing protocol.
        result[dataset] = dict(search=asdict(SearchConfig(**search)),
                               baseline={key: baseline[key] for key in sorted(BASELINE_FIELDS)})
    return result


def dataset_configuration(config, dataset):
    """Only NRT settings change; benchmark settings and outer splits stay shared."""
    selected = config['nrt_by_dataset'][dataset]
    result = deepcopy(config)
    result['nrt_search'] = deepcopy(selected['search'])
    result['nrt_baseline'].update(selected['baseline'])
    return result
