#!/usr/bin/env python3
"""
Launcher to use the mechanism of R. McKenna on the adult dataset.

Created from the match3.py file of the original repository:
https://github.com/usnistgov/PrivacyEngCollabSpace/tree/master/tools
de-identification/Differential-Privacy-Synthetic-Data-Challenge-Algorithms/
rmckenna

-----
Nampoina Andriamilanto <tompo.andri@gmail.com>
"""

import argparse
from functools import reduce
from pathlib import Path
from typing import Dict

import numpy as np
from loguru import logger
from scipy import sparse, optimize
from tensorflow_privacy.privacy.analysis.rdp_accountant import (
    compute_rdp, get_privacy_spent)

import mbi
from mbi import Domain, FactoredInference, Dataset

import matrix
from mechanism import Mechanism

DEFAULT_PARAMETERS = {
    'dataset': Path('data') / Path('adult-dataset-pickle.zip'),
    'domain': Path('data') / Path('adult-domain.json'),
    'epsilon': 1.0, 'delta': 2.2820544e-12,
    'save_path': Path('data') / Path('out.csv'),
    'backend': 'numpy'
}

# The elements of the noisy vector that exceed sigma multiplied by this
# multiplier will be compressed in a single value. See writeup.pdf.
SIGMA_MULTIPLIER = 3.0

FACTORED_INFERENCE_ITERATIONS = 500
CHECKPOINT_FILE_SUFFIX = '-checkpoint.csv'


def transform_data(data: mbi.Dataset, supports: Dict[str, np.ndarray]
                   ) -> mbi.Dataset:
    """Compress the domains of the dataset between round1 and round2.

    For a given attribute, we consider the values which noisy relative
    frequency are below a threshold as a single value. See writeup.pdf.

    Args:
        data: The dataset.
        supports: A dictionary mapping each attribute (name as str) to the
                  support values. It contains boolean values such that b_i
                  specifies whether the value v_i of the attribute is
                  significant (i.e., if its noisy relative frequency is above
                  sigma multiplied by SIGMA_MULTIPLIER).

    Returns:
        A dataset (mbi.Dataset) with the domains compressed.
    """
    compressed_dataframe = data.df.copy()
    compressed_domain_size = {}

    for col in data.domain:
        support = supports[col]

        # Infer the size of the new domain
        size = support.sum()
        compressed_domain_size[col] = int(size)
        if size < support.size:
            compressed_domain_size[col] += 1

        # Create the mapping dictionary.
        # If b_i is False, the value should not be compressed and held as is.
        # Otherwise, the value is set to the highest id (= size).
        mapping, value_id = {}, 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = value_id
                value_id += 1
        assert value_id == size

        # Update the values of this attribute in the new dataframe
        compressed_dataframe[col] = compressed_dataframe[col].map(mapping)

    # Create a new Dataset object with this dataframe and the new domain
    compressed_domain = Domain.fromdict(compressed_domain_size)
    return Dataset(compressed_dataframe, compressed_domain)


def reverse_data(data: mbi.Dataset, supports: Dict[str, np.ndarray]
                 ) -> mbi.Dataset:
    """Set back the domains in the initial format (uncompressed).

    Args:
        data: The dataset.
        supports: A dictionary mapping each attribute (name as str) to the
                  support values. It contains boolean values such that b_i
                  specifies whether the value v_i of the attribute is
                  significant (i.e., if its noisy relative frequency is above
                  sigma multiplied by SIGMA_MULTIPLIER).

    Returns:
        A dataset (mbi.Dataset) with the domains uncompressed.
    """
    uncompressed_dataframe = data.df.copy()
    initial_domain_size = {}

    # Uncompress each attribute
    for col in data.domain:
        support = supports[col]

        # Infer the maximum value, which is the index representing all the
        # compressed values
        maximum = support.sum()
        initial_domain_size[col] = int(support.size)

        # Get the ids of the values that were compressed and those that were
        # untouched (i.e., uncompressed)
        id_val_untouched = np.where(support)[0]
        id_val_compressed = np.where(~support)[0]

        # Create a mask which is True at i if the compressed value corresponds
        # to the maximum (i.e., they were compressed)
        mask = uncompressed_dataframe[col] == maximum

        # If some values were compressed (i.e., their index is the maximum),
        # replace these indices with a random index from the initial dataset.
        # The idea is to replace the compressed values with a random value in
        # the set of the compressed values.
        if id_val_compressed.size > 0:
            uncompressed_dataframe.loc[mask, col] = np.random.choice(
                id_val_compressed, mask.sum())

        uncompressed_dataframe.loc[~mask, col] = id_val_untouched[
            uncompressed_dataframe.loc[~mask, col]]

    initial_domain = Domain.fromdict(initial_domain_size)
    return Dataset(uncompressed_dataframe, initial_domain)


def moments_calibration(round1_l2_sens: float, round2_l2_sens: float,
                        eps: float, delta: float):
    """Infer the sigma parameter used in the Gaussian mechanism.

    Args:
        round1_l2_sens: L2 sensitivity of round1 queries.
        round2_l2_sens: L2 sensitivity of round2 queries.
        eps: The epsilon privacy parameter.
        delta: The delta privacy parameter.

    Returns:
        The sigma.
    """
    # works as long as eps >= 0.01; if larger, increase orders
    orders = range(2, 4096)

    def obj(sigma):
        rdp1 = compute_rdp(1.0, sigma/round1_l2_sens, 1, orders)
        rdp2 = compute_rdp(1.0, sigma/round2_l2_sens, 1, orders)
        rdp = rdp1 + rdp2
        privacy = get_privacy_spent(orders, rdp, target_delta=delta)
        return privacy[0] - eps + 1e-8

    low, high = 1.0, 1.0
    while obj(low) < 0:
        low /= 2.0
    while obj(high) > 0:
        high *= 2.0
    sigma = optimize.bisect(obj, low, high)
    assert obj(sigma) - 1e-8 <= 0, 'not differentially private'
    # true eps <= requested eps
    return sigma


class AdultMechanism(Mechanism):
    """The implementation of the mechanism for the adult dataset."""

    def __init__(self, dataset_path: Path, domain_info_path: Path,
                 backend: str, iters: int = 1000, warmup: bool = False):
        """Initialize the AdultMechanism.

        Args:
            dataset_path: The path to the dataset file (in csv format).
            domain_info_path: The path to the domain information file.
            backend: The backend to use (numpy or torch).
            iters: The number of iterations to execute.
            warmup: Whether we should set the FactoredInference engine up.
        """
        Mechanism.__init__(self, dataset_path, domain_info_path)
        self.backend = backend
        self.iters = iters
        self.warmup = warmup
        self.supports = None
        self.engine = None
        self.measurements = []
        self.round1 = []
        self.round2 = []

    def setup(self):
        """Do any setup needed to run the algorithm here."""
        # The round1 contains the 1-way marginals
        self.round1 = list(self.domain.attrs)

        # The round2 contains a carefully chosen set of 2 and 3-way marginals
        # (see the select-queries notebook)
        self.round2 = [('age', 'marital-status'),
                       ('workclass', 'occupation'),
                       ('fnlwgt', 'native-country'),
                       ('education', 'education-num'),
                       ('education', 'occupation'),
                       ('education-num', 'native-country'),
                       ('marital-status', 'relationship'),
                       ('occupation', 'sex'),
                       ('occupation', 'hours-per-week'),
                       ('relationship', 'sex'),
                       ('relationship', 'income'),
                       ('race', 'native-country'),
                       ('capital-gain', 'income'),
                       ('capital-loss', 'income')]

        self.round2 += [('relationship', 'age'),
                        ('marital-status', 'relationship', 'age'),
                        ('workclass', 'education'),
                        ('education', 'hours-per-week'),
                        ('sex', 'hours-per-week'),
                        ('occupation', 'workclass', 'education'),
                        ('occupation', 'education', 'hours-per-week'),
                        ('occupation', 'sex', 'hours-per-week'),
                        ('marital-status', 'sex'),
                        ('relationship', 'marital-status', 'sex'),
                        ('relationship', 'occupation'),
                        ('sex', 'relationship', 'occupation'),
                        ('race', 'fnlwgt'),
                        ('education-num', 'fnlwgt'),
                        ('native-country', 'race', 'fnlwgt'),
                        ('native-country', 'education-num', 'fnlwgt')]

    def measure(self):
        """Load the data and measure things about it.

        Save the measurements taken, but do not save the data. This is the only
        function that needs to be vetted for privacy.
        """
        data = self.load_data()
        logger.debug('The data was correctly loaded, beginning the measures')

        # The round1 and round2 measurements will be weighted to have L2
        # sensitivity 1
        sigma = moments_calibration(1.0, 1.0, self.epsilon, self.delta)
        logger.info(f'Noise level (sigma) set to {sigma}')

        logger.info('Beginning of the round 1')

        # Initialize the weights
        weights = np.ones(len(self.round1))
        weights /= np.linalg.norm(weights)  # Now has L2 norm = 1
        logger.debug(f'Initialized the weights of round 1 as {weights}')

        # First round (the 1-way marginals)
        supports = {}
        for col, wgt in zip(self.round1, weights):
            # Noise-addition step using the Gaussian mechanism on the 1-way
            # marginals
            proj = (col,)
            logger.debug(f'Projection on the attributes {proj}')
            hist = data.project(proj).datavector()
            logger.debug(f'  Histogram (size={hist.size}): {hist}')
            noise = sigma * np.random.randn(hist.size)
            noisy_hist = wgt * hist + noise
            logger.debug(f'  Noisy histogram: {noisy_hist}')

            # Post-processing step which consists into compressing domains.
            # The values that have a noisy relative frequency above the
            # threshold function of sigma (the standard deviation) will be
            # represented by a single value in the final representation.
            support = (noisy_hist >= SIGMA_MULTIPLIER * sigma)
            supports[col] = support
            logger.debug(f'  Support: {support}')

            # If all the values exceed the threshold
            if support.sum() == noisy_hist.size:
                final_noisy_hist = noisy_hist
                diagonal_matrix = matrix.Identity(noisy_hist.size)
            else:
                # Generate a new histogram with the values that we hold (these
                # that are above the threshold) and the sum of the remaining
                # relative frequencies.
                # Example: Starting with the histogram [0.2, 0.6, 0.1, 0.1] and
                #          the support [True, True, False, False] gives
                #          [0.2, 0.6, 0.2] as a result.
                final_noisy_hist = np.append(
                    noisy_hist[support], noisy_hist[~support].sum())
                ones_vector = np.ones(final_noisy_hist.size)
                ones_vector[-1] = 1.0 / np.sqrt(
                    noisy_hist.size - final_noisy_hist.size + 1.0)
                final_noisy_hist[-1] /= np.sqrt(
                    noisy_hist.size - final_noisy_hist.size + 1.0)
                diagonal_matrix = sparse.diags(ones_vector)

            weighted_noisy_histogram = final_noisy_hist/wgt
            weight_inverse = 1.0/wgt
            logger.debug(f'  Attributes: {proj}')
            logger.debug(f'  Diagonal matrix: {diagonal_matrix}')
            logger.debug(f'  Final noisy histogram: {final_noisy_hist}')
            logger.debug('  Weighted noisy histogram: '
                         f'{weighted_noisy_histogram}')
            logger.debug(f'  Weight inverse: {weight_inverse}')

            self.measurements.append(
                (diagonal_matrix, weighted_noisy_histogram, weight_inverse,
                 proj))

        logger.info('End of the round 1 and beginning of the round 2')

        self.supports = supports

        # Perform round 2 measurements over compressed domain
        data = transform_data(data, supports)
        self.domain = data.domain

        weights = np.ones(len(self.round2))
        weights /= np.linalg.norm(weights)  # Now has L2 norm = 1
        logger.debug(f'Initialized the weights of round 2 as {weights}')

        for proj, wgt in zip(self.round2, weights):
            # Noise-addition step using the Gaussian mechanism on the 2/3-way
            # marginals
            hist = data.project(proj).datavector()
            Q = matrix.Identity(hist.size)

            noise = sigma * np.random.randn(Q.shape[0])
            noisy_hist = wgt * Q.dot(hist) + noise
            weighted_noisy_histogram = noisy_hist/wgt
            weight_inverse = 1.0/wgt

            logger.debug(f'  Attributes: {proj}')
            logger.debug(f'  Q: {Q}')
            logger.debug(f'  Noisy histogram: {noisy_hist}')
            logger.debug('  Weighted noisy histogram: '
                         f'{weighted_noisy_histogram}')
            logger.debug(f'  Weight inverse: {weight_inverse}')

            self.measurements.append(
                (Q, weighted_noisy_histogram, weight_inverse, proj))

    def postprocess(self):
        """Postprocess step.

        Post-process the measurements taken into a synthetic dataset over
        discrete attributes.
        """
        # Set the engine (FactoredInference)
        self.engine = FactoredInference(
            self.domain, iters=FACTORED_INFERENCE_ITERATIONS, log=True,
            warm_start=True, backend=self.backend)
        engine = self.engine
        callback = mbi.callbacks.Logger(engine)

        if self.warmup:
            engine._setup(self.measurements, None)

            oneway = {}
            for i in range(len(self.round1)):
                attribute = self.round1[i]
                noisy_hist = self.measurements[i][1]
                noisy_hist = np.maximum(noisy_hist, 1)
                noisy_hist /= noisy_hist.sum()

                # The type of Factor used depends on the backend
                if self.backend == 'torch':
                    oneway[attribute] = mbi.torch_factor.Factor(
                        self.domain.project(attribute), noisy_hist)
                else:
                    oneway[attribute] = mbi.Factor(
                        self.domain.project(attribute), noisy_hist)

            marginals = {}
            for clique in engine.model.cliques:
                marginals[clique] = reduce(
                    lambda x, noisy_hist: x * noisy_hist,
                    [oneway[attribute] for attribute in clique])

            theta = engine.model.mle(marginals)
            engine.potentials = theta
            # EDIT: We replaced the function named belief_prop_fast by
            # belief_propagation. Thanks to Pascale-Jade Vallot.
            engine.marginals = engine.model.belief_propagation(theta)

        # Create the path to the checkpoint file where we will save the current
        # synthetic dataset regularly
        checkpoint_path = self.save_path.with_name(
            self.save_path.stem + CHECKPOINT_FILE_SUFFIX)

        # Execute FACTORED_INFERENCE_ITERATIONS iterations enough time to reach
        # the self.iters number of total iterations.
        # Every FACTORED_INFERENCE_ITERATIONS iterations, we display the
        # current state.
        for i in range(self.iters // FACTORED_INFERENCE_ITERATIONS):
            engine.infer(self.measurements, engine='MD', callback=callback)

            # Every four prints, save the current synthetic dataset
            if i % 4 == 3:
                synthetic = engine.model.synthetic_data()
                self.synthetic = reverse_data(synthetic, self.supports)
                self.transform_domain()
                self.synthetic.to_csv(checkpoint_path, index=False)

        # Remove the checkpoint file if it exists
        checkpoint_path.unlink(missing_ok=True)

        # Set the final synthetic dataset
        self.synthetic = engine.model.synthetic_data()
        self.synthetic = reverse_data(self.synthetic, self.supports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help='path to dataset csv file')
    parser.add_argument('--domain', help='path to domain json file')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--save-path', help='path to save synthetic data to')
    parser.add_argument('--backend', help='the backend to use',
                        choices=['numpy', 'torch'])

    parser.set_defaults(**DEFAULT_PARAMETERS)
    args = parser.parse_args()

    ITERS = 7500
    if args.epsilon >= 4.0:
        ITERS = 10000

    mech = AdultMechanism(args.dataset, args.domain, args.backend, iters=ITERS,
                          warmup=True)

    mech.run(args.epsilon, args.save_path, args.delta)
