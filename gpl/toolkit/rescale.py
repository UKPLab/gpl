from cmath import log
import os
import shutil
import logging
logger = logging.getLogger(__name__)


def rescale_fn(margin, org_min, org_max, new_min, new_max):
    if org_max == 0 or org_min == 0:
        pass
    elif margin > 0:
        org_min = 0
        new_min = 0
    else:
        assert margin <= 0
        org_max = 0
        new_max = 0

    new_margin = new_min + (margin - org_min) * (new_max - new_min) / (org_max - org_min)
    return new_margin


def rescale_gpl_training_data(data_dir: str, new_min: float, new_max: float, suffix='rescaled'):
    gpl_training_data_name = 'gpl-training-data.tsv'
    assert gpl_training_data_name in os.listdir(data_dir)
    fpath = os.path.join(data_dir, gpl_training_data_name)
    margins = []
    with open(fpath, 'r') as f:
        for i, line in enumerate(f):
            margin = float(line.strip().split('\t')[-1])
            margins.append(margin)
    org_min, org_max = min(margins), max(margins)  # TODO: Try points at 5% and 95% percentiles
    assert org_min != org_max

    frescaled = f'gpl-training-data.{suffix}.tsv'
    fpath_rescaled = os.path.join(data_dir, frescaled)
    if os.path.exists(fpath_rescaled):
        logger.info('Found the rescaled data has already existed. Escaped rescaling')
        return frescaled

    with open(fpath, 'r') as fin, open(fpath_rescaled, 'w') as fout:
        for line in fin:
            items = line.strip().split('\t')
            margin = float(items[-1])

            # rescaling:            
            new_margin = rescale_fn(margin, org_min, org_max, new_min, new_max)

            items[-1] = str(new_margin)
            line = '\t'.join(items) + '\n'
            fout.write(line)

    logger.info(f'Rescaled the pseudo labels to range [{new_min}, {new_max}] from the original range [{org_min}, {org_max}]. The output is {fpath_rescaled}')
    return frescaled

