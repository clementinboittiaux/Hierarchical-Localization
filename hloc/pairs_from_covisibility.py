import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from . import logger
from .utils.read_write_model import read_model


def main(model, output, num_matched, query_list: Path = None, db_model: Path = None):
    logger.info('Reading the COLMAP model...')
    cameras, images, points3D = read_model(model)

    if query_list is not None:
        query_names = query_list.read_text().splitlines()
    else:
        query_names = [image.name for image in images.values()]

    if db_model is not None:
        db_names = [image.name for image in read_model(db_model)[1].values()]
    else:
        db_names = [image.name for image in images.values()]

    logger.info('Extracting image pairs from covisibility info...')
    pairs = []
    for image_id, image in tqdm(images.items()):

        if image.name not in query_names:
            continue

        matched = image.point3D_ids != -1
        points3D_covis = image.point3D_ids[matched]

        covis = defaultdict(int)
        for point_id in points3D_covis:
            for image_covis_id in points3D[point_id].image_ids:
                if image_covis_id != image_id:
                    covis[image_covis_id] += 1

        if len(covis) == 0:
            logger.info(f'Image {image_id} does not have any covisibility.')
            continue

        covis_ids = np.array([i for i in covis if images[i].name in db_names])
        covis_num = np.array([covis[i] for i in covis_ids])

        if len(covis_ids) <= num_matched:
            top_covis_ids = covis_ids[np.argsort(-covis_num)]
        else:
            # get covisible image ids with top k number of common matches
            ind_top = np.argpartition(covis_num, -num_matched)
            ind_top = ind_top[-num_matched:]  # unsorted top k
            ind_top = ind_top[np.argsort(-covis_num[ind_top])]
            top_covis_ids = [covis_ids[i] for i in ind_top]
            assert covis_num[ind_top[0]] == np.max(covis_num)

        for i in top_covis_ids:
            pair = (image.name, images[i].name)
            pairs.append(pair)

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--num_matched', required=True, type=int)
    args = parser.parse_args()
    main(**args.__dict__)
