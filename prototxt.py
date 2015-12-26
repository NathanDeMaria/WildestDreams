import os
import logging
from jinja2 import Template
from global_params import BATCH_SIZE, DEPLOY_PROTOTXT, LMDB_NAME, PIXELS, SOLVER_PROTOTXT, TEAMS, TRAIN_PROTOTXT


def _rewrite_template(base_file, **kwargs):
    """
    Fills one of the template files and writes the filled in template to
    a local file with matching name
    :param base_file: name of file to fill template for
    :param kwargs: named args to pass to render
    """
    try:
        path, file_name = os.path.split(base_file)
        template_file = os.path.join(path, 'templates', file_name)
        with open(template_file) as f:
            raw_template = f.read()
    except OSError as e:
        logging.ERROR("Template for {0} not found".format(base_file))
        raise e

    template = Template(raw_template)
    rendered = template.render(**kwargs)
    with open(base_file, 'w') as f:
        f.write(rendered)


def write_files(solver_prototxt=SOLVER_PROTOTXT, train_prototxt=TRAIN_PROTOTXT, deploy_prototxt=DEPLOY_PROTOTXT,
                pixels=PIXELS, lmdb_name=LMDB_NAME, teams=TEAMS, batch_size=BATCH_SIZE):
    """
    Write some caffe templates
    :param solver_prototxt: solver.prototxt file
    :param train_prototxt: train.prototxt file
    :param deploy_prototxt: deploy.prototxt file
    :param pixels: number of pixels we rescaled image to
    :param lmdb_name: name of lmdb file data is saved to
    :param teams: list of teams
    :param batch_size: size of a batch
    """
    _rewrite_template(solver_prototxt, train_prototxt=train_prototxt, solver_mode='GPU')
    _rewrite_template(deploy_prototxt, pixels=pixels, num_output=len(teams))
    _rewrite_template(train_prototxt, pixels=pixels, lmdb=lmdb_name, num_output=len(teams), batch_size=batch_size)
