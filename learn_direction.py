import os
import logging
import datetime

import numpy
from sklearn.linear_model import LogisticRegression
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--latent_dir', type=str, required=True)
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--direction_dir', type=str, required=True)


def main(opts):
    handlers = [logging.FileHandler(
        os.path.join(opts.log_dir, 'learn_direction ' + str(datetime.datetime.now()) + '.log')),
                logging.StreamHandler()]
    logging.basicConfig(handlers=handlers,
                        format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    latent_dir = opts.latent_dir
    direction_dir = opts.direction_dir
    latents = np.load(os.path.join(latent_dir, 'latent.npy'))

    for filename in os.listdir(os.path.join(latent_dir, 'label')):
        attr_name = os.path.splitext(filename)[0]
        labels = np.load(os.path.join(latent_dir, 'label', filename))
        if labels.sum() == 0 or labels.sum() == len(labels):
            direction = np.zeros((18, 512))
            np.save(os.path.join(direction_dir, filename), direction)
            logger.info(f"{attr_name} has all the same labels")
        latents = latents.reshape((-1, 512*18))
        clf = LogisticRegression(class_weight='balanced', max_iter=500).fit(latents, labels)
        predictions = clf.predict(latents)
        acc = 1 - numpy.abs(predictions - labels).sum() / len(labels)
        logger.info(f"accuracy on {attr_name} is {acc}")
        direction = clf.coef_.reshape((18, 512))
        direction /= np.sqrt((direction * direction).sum())  # convert to unit vector
        os.makedirs(direction_dir, exist_ok=True)
        np.save(os.path.join(direction_dir, filename), direction)

# def main(opts):
#     latents_dir = '/home/hs3374/dlcv_final/latents'
#     direction_dir = '/home/hs3374/dlcv_final/directions'
#     handlers = [logging.FileHandler(
#         os.path.join('/home/hs3374/dlcv_final/log', 'learn_direction ' + str(datetime.datetime.now()) + '.log')),
#                 logging.StreamHandler()]
#     logging.basicConfig(handlers=handlers,
#                         format='%(asctime)s %(message)s')
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     for filename in os.listdir(os.path.join(latents_dir, 'latent')):
#         if os.path.exists(os.path.join(direction_dir, filename)):
#             continue
#         attr_name = os.path.splitext(filename)[0]
#         latents = np.load(os.path.join(latents_dir, 'latent', filename))
#         labels = np.load(os.path.join(latents_dir, 'label', filename))
#         if labels.sum() == 0 or labels.sum() == len(labels):
#             direction = np.zeros((18, 512))
#             np.save(os.path.join(direction_dir, filename), direction)
#             logger.info(f"{attr_name} has all the same labels.")
#             continue
#         latents = latents.reshape((-1, 512*18))
#         clf = LogisticRegression(class_weight='balanced', max_iter=500).fit(latents, labels)
#         predictions = clf.predict(latents)
#         acc = 1 - numpy.abs(predictions - labels).sum() / len(labels)
#         logger.info(f"accuracy on {attr_name} is {acc}")
#         direction = clf.coef_.reshape((18, 512))
#         direction /= np.sqrt((direction * direction).sum())  # convert to unit vector
#         np.save(os.path.join(direction_dir, filename), direction)

if __name__ == '__main__':
    opts = parser.parse_args()
    main(opts)