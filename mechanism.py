#!/usr/bin/env python3
"""
The implementation of the Mechanism class.

Created from the mechanism.py file of the original repository:
https://github.com/usnistgov/PrivacyEngCollabSpace/tree/master/tools
de-identification/Differential-Privacy-Synthetic-Data-Challenge-Algorithms/
rmckenna

-----
Nampoina Andriamilanto <tompo.andri@gmail.com>
"""

import json
import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

import mbi
from mbi import Dataset, Domain

# The attributes that are discretized
# Format: index => (lower bound included, higher bound excluded)
DISCRETIZED_ATTRIBUTES = {
    'age': {0: (17, 26), 1: (26, 62), 2: (62, 100)},
    'fnlwgt': {0: (10000, 100000), 1: (100000, 200000), 2: (200000, 300000),
               3: (300000, 400000), 4: (400000, 500000), 5: (500000, 1500000)},
    'capital-gain': {0: (0, 5000), 1: (5000, 10000), 2: (10000, 20000),
                     3: (20000, 100000)},
    'capital-loss': {0: (0, 1000), 1: (1000, 2000), 2: (2000, 3000),
                     3: (3000, 4000), 5: (4000, 5000)},
    'hours-per-week': {0: (0, 25), 1: (25, 50), 2: (50, 75),
                       3: (75, 100)}
}


class Mechanism:
    """The Mechanism class.

    This class is a template for a mechanism with all the boilerplate code
    already implemented. Subclasses should implement three functions:
        setup, measure, and postprocess

    measure is the only function that is allowed to look at the data, and it
    must be privacy-vetted. All other code should not need to be checked with
    very much scrutiny.
    """

    def __init__(self, dataset_path: Path, domain_info_path: Path):
        """Initialize the mechanism.

        Args:
            dataset_path: The path to the dataset file (in csv format).
            domain_info_path: The path to the domain information file.
        """
        # Set the dataset path
        self.dataset_path = dataset_path
        logger.info(f'Set the dataset path: {self.dataset_path}')

        # Load the domain information
        # Format: attribute => list[attr. values]
        self.domain_info = json.load(open(domain_info_path, 'r'))
        logger.info(f'Loaded the domain from {domain_info_path}')
        logger.debug(self.domain_info)

        # Infer the domain sizes (used for the domain object)
        # Format: attribute => size of the domain of the attribute
        domain_sizes = {}
        for attribute in self.domain_info:
            domain_sizes[attribute] = len(self.domain_info[attribute])
        self.domain = Domain.fromdict(domain_sizes)
        logger.debug(f'The domain size of each attribute: {self.domain}')

        # This will hold the initial order of the columns and is set in
        # load_data()
        self.column_order = None

        # The synthetic dataset. Note that depending on the moment, this
        # attribute can be of type mbi.Dataset or a pd.DataFrame
        self.synthetic = None

        # The privacy parameters that will be set in run()
        self.epsilon = None
        self.delta = None
        self.save_path = None

    def setup(self):
        """Do any setup needed to run the algorithm here."""

    def load_data(self, dataset_path: Optional[Path] = None) -> mbi.Dataset:
        """Load the data and discretize the integer/float attributes.

        Args:
            dataset_path: The path to the dataset file (in csv format).

        Returns:
            The loaded dataset (type mbi.Dataset).
        """
        # Load the dataset
        if dataset_path is None:
            dataset_path = self.dataset_path

        dataset = pd.read_pickle(dataset_path)
        logger.info(f'Loaded {len(dataset)} rows from {dataset_path}')

        # Backup the order of the columns (without the unwated attributes)
        self.column_order = dataset.columns
        logger.debug(f'The initial column order: {self.column_order}')

        # Build the set of each attribute type
        attributes = set(self.domain_info.keys())
        continuous_attributes = set(DISCRETIZED_ATTRIBUTES.keys())
        categorical_attributes = attributes.difference(continuous_attributes)
        logger.info(f'We have {len(attributes)} attributes, divided into '
                    f'{len(continuous_attributes)} continuous attributes and '
                    f'{len(categorical_attributes)} categorical attributes')
        logger.debug(f'Continuous attributes: {continuous_attributes}')
        logger.debug(f'Categorical attributes: {categorical_attributes}')

        # Map the values of the categorical attributes to their indices.
        # Format of mapping: value => index
        for attribute in categorical_attributes:
            values = self.domain_info[attribute]
            mapping = dict(zip(values, range(len(values))))
            dataset[attribute] = dataset[attribute].map(mapping)

        # For each continuous attribute, create the mapping into the bins
        for attribute, bin_id__bins in DISCRETIZED_ATTRIBUTES.items():

            def cont_attr_mapper(value: Any) -> int:
                """Map the value of a continuous attribute to its bin index.

                Args:
                    value: The continuous value to map to a bin index.

                Returns:
                    The index of the bin in which the value falls in.
                """
                for bin_id, bin_range in bin_id__bins.items():
                    lower_bound, higher_bound_ex = bin_range
                    if lower_bound <= value < higher_bound_ex:
                        return bin_id
                raise ValueError(f'The value {value} of {attribute} does not '
                                 'fall within a bin!')

            dataset[attribute] = dataset[attribute].map(cont_attr_mapper)

        return Dataset(dataset, self.domain)

    def measure(self):
        """Load the data and measure things about it.

        Save the measurements taken, but do not save the data. This is the only
        function that needs to be vetted for privacy.
        """

    def postprocess(self):
        """Postprocess step.

        Post-process the measurements taken into a synthetic dataset over
        discrete attributes.
        """

    def transform_domain(self):
        """Set the synthetic dataset to the initial domain.

        Convert the synthetic discrete data back to the original domain and add
        any missing columns with a default value. The synthetic dataset is then
        stored in self.synthetic.
        """
        # Get the synthetic dataset
        synthetic_dataset = self.synthetic.df

        # Build the set of each attribute type
        attributes = set(self.domain_info.keys())
        continuous_attributes = set(DISCRETIZED_ATTRIBUTES.keys())
        categorical_attributes = attributes.difference(continuous_attributes)

        # Map back the values of the categorical attributes from their indices.
        # Format of mapping: index => value
        for attribute in categorical_attributes:
            values = self.domain_info[attribute]
            mapping = dict(zip(range(len(values)), values))
            synthetic_dataset[attribute] = synthetic_dataset[attribute].map(
                mapping)

        # For each continuous attribute, create the mapping into the bins
        for attribute, bin_id__bins in DISCRETIZED_ATTRIBUTES.items():

            def cont_attr_mapper(bind_id) -> Any:
                """Map the bin index of a continuous attribute to its value.

                Args:
                    The index of the bin in which the value falls in.

                Returns:
                    value: A random continuous value that falls within the bin.
                """
                lower_bound, higher_bound_excld = bin_id__bins[bind_id]
                return random.randrange(lower_bound, higher_bound_excld)

            synthetic_dataset[attribute] = synthetic_dataset[attribute].map(
                cont_attr_mapper)

        # Set back to the right column order
        self.synthetic = synthetic_dataset[self.column_order]

    def run(self, epsilon: float, save_path: Path,
            delta: float = 2.2820610e-12) -> pd.DataFrame:
        """Run the mechanism at the given privacy level.

        Args:
            epsilon: The privacy budget.
            save: The location where to save the synthetic data.
            delta: The privacy parameter.

        Returns:
            The synthetic data in the same format as original data.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.save_path = save_path
        self.setup()
        self.measure()
        self.postprocess()
        self.transform_domain()
        self.synthetic.to_csv(save_path, index=False)
        return self.synthetic
