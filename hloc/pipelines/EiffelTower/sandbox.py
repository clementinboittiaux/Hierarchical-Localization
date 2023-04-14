import h5py
import argparse
import numpy as np
from pathlib import Path
from pprint import pformat

import pycolmap

from ..Cambridge.utils import create_query_list_with_intrinsics, evaluate
from ... import extract_features, match_features, triangulation, match_dense
from ... import logger
from ... import pairs_from_covisibility, pairs_from_retrieval, localize_sfm
from ...utils.parsers import parse_image_list


def create_query_list(model: Path, queries: Path):
    reconstruction = pycolmap.Reconstruction(model)
    queries.parent.mkdir(parents=True, exist_ok=True)
    with open(queries, 'w') as q:
        for image in reconstruction.images.values():
            if image.name[:4] == '2015':  # images of the 2015 visit are the query images
                q.write(f'{image.name}\n')


def create_db_sfm(model: Path, queries: Path, output: Path):
    '''Filter the model to discard the query images and keep only the reference images.
    '''
    reconstruction = pycolmap.Reconstruction(model)
    query_names = parse_image_list(queries)
    query_ids = [image.image_id for image in reconstruction.images.values() if image.name in query_names]
    for query_id in query_ids:
        reconstruction.deregister_image(query_id)
    output.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output)


def pairs_from_brut_force(matches_path: Path, pairs_path: Path, num_loc: int):
    with h5py.File(matches_path, 'r', libver='latest') as r, open(pairs_path, 'w') as w:
        for query_name, db_group in r.items():
            db_names, num_matches = [], []
            for db_name, db_matches in db_group.items():
                db_names.append(db_name)
                num_matches.append(np.sum(db_matches['matches0'][()] != -1))
            top_db_names = np.array(db_names)[np.argsort(num_matches)[::-1]]
            for db_name in top_db_names[:num_loc]:
                w.write(f'{query_name} {db_name}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='datasets/eiffeltowerresized',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs/eiffeltowersandbox',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_covis', type=int, default=20,
                        help='Number of image pairs for SfM, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=10,
                        help='Number of image pairs for loc, default: %(default)s')
    args = parser.parse_args()

    # set up the paths
    dataset = args.dataset
    images = dataset / 'global' / 'images'
    global_sfm = dataset / 'global' / 'sfm'  # the global SfM model with both reference and query images

    outputs = args.outputs  # where everything will be saved
    query_list = outputs / 'list_query.txt'  # where to save the list of query images
    query_list_with_intrinsics = outputs / 'list_query_with_intrinsics.txt'  # list of query images with intrinsics
    db_sfm = outputs / 'sfm_db'  # where to save the global SfM filtered with only the reference images
    reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
    sfm_pairs = outputs / f'pairs-db-covis{args.num_covis}.txt'  # top-k most covisible in dataset model
    loc_pairs = outputs / f'pairs-query-netvlad{args.num_loc}.txt'  # top-k retrieved by NetVLAD
    results = outputs / f'EiffelTower_hloc_superpoint+superglue_netvlad{args.num_loc}.txt'

    # list the standard configurations available
    logger.info(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    logger.info(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    # matcher_conf = match_dense.confs['loftr_aachen']

    # set up query lists and database model from the Eiffel Tower dataset
    # logger.info(f'Create lists of query images: {query_list}.')
    # create_query_list(global_sfm, query_list)

    # logger.info(f'Create list of query images with intrinsics: {query_list_with_intrinsics}.')
    # create_query_list_with_intrinsics(global_sfm, query_list_with_intrinsics, list_file=query_list, ext='.txt')

    # logger.info(f'Filter global SfM model to keep only the reference images: {db_sfm}.')
    # create_db_sfm(global_sfm, query_list, db_sfm)

    # features = extract_features.main(feature_conf, images, outputs)
    features = outputs / f"{feature_conf['output']}.h5"
    #
    # pairs_from_covisibility.main(
    #     db_sfm, sfm_pairs, num_matched=args.num_covis)
    # sfm_matches = match_features.main(
    #     matcher_conf, sfm_pairs, feature_conf['output'], outputs)
    #
    # triangulation.main(
    #     reference_sfm,
    #     db_sfm,
    #     images,
    #     sfm_pairs,
    #     features,
    #     sfm_matches)

    # global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    # pairs_from_retrieval.main(
    #     global_descriptors, loc_pairs, args.num_loc,
    #     query_list=query_list, db_model=reference_sfm)

    # pairs_from_covisibility.main(global_sfm, loc_pairs, args.num_loc, query_list=query_list, db_model=reference_sfm)

    # loc_pairs = outputs / 'vocab' / 'loc_pairs_sucre_360.txt'

    # loc_matches = match_features.main(
    #     matcher_conf, loc_pairs, feature_conf['output'], outputs)
    loc_matches = outputs / 'feats-superpoint-n4096-r1024_matches-superglue_pairs-query-netvlad10.h5'

    # pairs_from_covisibility.main(
    #     db_sfm, sfm_pairs, num_matched=args.num_covis)
    # features, sfm_matches = match_dense.main(matcher_conf, sfm_pairs, images,
    #                                          outputs, max_kps=8192,
    #                                          overwrite=False)
    #
    # triangulation.main(
    #     reference_sfm,
    #     db_sfm,
    #     images,
    #     sfm_pairs,
    #     features,
    #     sfm_matches)
    #
    # pairs_from_covisibility.main(global_sfm, loc_pairs, args.num_loc, query_list=query_list, db_model=reference_sfm)
    #
    # features, loc_matches = match_dense.main(
    #     matcher_conf, loc_pairs, images, outputs, features=features, max_kps=None,
    #     matches=sfm_matches)

    localize_sfm.main(
        reference_sfm,
        query_list_with_intrinsics,
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False)  # not required with SuperPoint+SuperGlue

    evaluate(
        global_sfm, results,
        query_list, ext='.txt')
