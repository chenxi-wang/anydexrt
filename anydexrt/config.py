"""Config loading.
Author: Chenxi Wang, Ying Feng
"""
import json
from pathlib import Path
from types import SimpleNamespace


def load_config(path):
    path = Path(path).resolve()
    with path.open() as f:
        config = json.load(f)
    n = config['finger_num']
    shape = config['joint']['shape']
    if shape[0] != n or shape[0] * shape[1] != config['joint']['num']:
        raise ValueError('Inconsistent finger count and padded joint shape')
    mapping = config['joint']['hand2manus_mapping']
    if set(map(int, mapping)) != set(range(n)) or len(set(mapping.values())) != n:
        raise ValueError('hand2manus_mapping must map each robot finger uniquely')
    if not set(mapping.values()) <= set(range(5)):
        raise ValueError('MANUS finger indices must be between 0 and 4')
    for finger, glove in mapping.items():
        if config['joint']['manus2hand_mapping'].get(str(glove)) != int(finger):
            raise ValueError('Finger mappings must be inverses')
    for name in ['limit', 'anchor_bend_limit', 'anchor_lateral_limit']:
        for bound in ['lower', 'upper']:
            values = config['joint'][name][bound]
            if len(values) != n or any(len(row) != shape[1] for row in values):
                raise ValueError(f'{name}/{bound} must match joint shape')
    urdf = Path(config['urdf_path'])
    if not urdf.is_absolute():
        urdf = path.parent.parent / urdf
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    config['urdf_path'] = str(urdf.resolve())
    return config


def hand_spec(config):
    """Only metadata is needed for offline model training, not a visualizer."""
    return SimpleNamespace(
        finger_num=config['finger_num'],
        hand2manus={int(k): v for k, v in config['joint']['hand2manus_mapping'].items()},
    )
