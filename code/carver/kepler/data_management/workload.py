# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates query workloads and plans based on collected data.

The DatabaseSimulator enables rapid iteration on and evaluation of machine
learning ideas and experiments. The KeplerPlanDiscoverer and WorkloadGenerator
provide support structure to generate training data using the DatabaseSimulator.

The KeplerPlanDiscoverer encapsulates identifying plan candidates.

The WorkloadGenerator constructs Workloads from the universe of possible
parameter bindings, as defined by the backing training data files.

```
Typical usage:

plan_ids = KeplerPlanDiscoverer(query_execution_data)

generator = WorkloadGenerator(query_execution_data)
workload = generator.random_sample(10)

# See DatabaseSimulator for integrated usage.
"""

import dataclasses
import itertools
import random
from typing import Any, List, Optional, Tuple
import json

from kepler.data_management import database_simulator
# TODO(b/199162711): Transition this script and downstream analysis scripts to a
# structured format instead of using _NAME_DELIMITER.
_NAME_DELIMITER = "####"


class KeplerPlanDiscoverer:
  """Discover the set of query plans to use for this query.

  Attributes:
    plan_ids: List of plan ids representing the identified query plan
      candidates.
  """

  def __init__(self,
               query_execution_data: Optional[Any] = None,
               query_execution_metadata: Optional[Any] = None):
    """Extracts plan ids from a provided source.

    The candidate_plan_cover provides a filtered subset of all the plans
    executed. If not available, the candidate plan set can be initialized to all
    plans using query_execution_data.

    Args:
      query_execution_data: Execution data structure that defines all the known
        information regarding query plans, parameter bindings, and latencies.
        The format is a series of nested dicts, typically parsed from a JSON
        file.
      query_execution_metadata: Metadata from query execution that is expected
        to contain a "plan_cover" entry.

    Raises:
      ValueError: If both or neither of query_execution_data and
        query_execution_metadata are provided.
    """
    if (query_execution_data is None and
        query_execution_metadata is None) or (query_execution_data and
                                              query_execution_metadata):
      raise ValueError(
          "Exactly one of query_execution_data and query_execution_metadata "
          "must be provided.")

    if query_execution_data:
      data_mapping = query_execution_data[next(iter(query_execution_data))]
      data_point = data_mapping[next(iter(data_mapping))]
      self.plan_ids = list(range(len(data_point["results"])))

    if query_execution_metadata:
      metadata_mapping = query_execution_metadata[next(
          iter(query_execution_metadata))]
      self.plan_ids = metadata_mapping["plan_cover"]


@dataclasses.dataclass
class QueryInstance:
  """An instantiation of the query template.

  Attributes:
    execution_frequency: How frequently this instance is executed in a workload.
    parameters: List of parameter bindings that define this instance.
  """
  execution_frequency: int
  parameters: List[str]


@dataclasses.dataclass
class Workload:
  """A set of parameter bindings and execution frequencies describing a workload.

  Workload parameter bindings simulate parameters recorded in a production
  system query log. Therefore, they are unlabeled. Labels, ie query execution
  latency data, must be obtained via the DatabaseSimulator.

  Attributes:
    query_id: The identifier for the query template.
    query_log: List of query instances to be executed together as a workload.
  """
  query_id: str
  query_log: List[QueryInstance]


class WorkloadGenerator:
  """Generate a workload on interest from the universe of possible parameter bindings.

  Attributes:
    parameter_count: The number of parameters in the query template.
    workload_pool_size: The number of distinct parameter bindings for each
      execution data was provided. All workloads are generated as subsets of
      this pool.
  """

  def __init__(self, query_execution_data: Any, seed: int = 0):
    """Build workload pool from provided execution data.

    Args:
      query_execution_data: Execution data structure that defines all the known
        information regarding query plans, parameter bindings, and latencies.
        The format is a series of nested dicts, typically parsed from a JSON
        file.
      seed: Seed to provide random library to reproduce samples.
    """

    assert len(query_execution_data) == 1, "Unexpected data format."
    self._seed = seed
    random.seed(self._seed)
    self._query_id = next(iter(query_execution_data))

    data_mapping = query_execution_data[self._query_id]
    self._parameters = []
    for parameters_as_key, results in data_mapping.items():
      # Ignore parameters that have default plan timed out.
      if "default_timed_out" in results["results"]:
        continue
      # We rsplit since there is a param value with c#; in general our delimiter
      # of #### won't work if there are both params that begin and end with #.
      self._parameters.append(parameters_as_key.rsplit(_NAME_DELIMITER))

  @property
  def parameter_count(self):
    return len(self._parameters[0])

  @property
  def workload_pool_size(self):
    return len(self._parameters)

  def all(self) -> Workload:
    """Generates a Workload of all parameters."""
    query_log = []
    for parameters in self._parameters:
      query_log.append(
          QueryInstance(execution_frequency=1, parameters=parameters))

    return Workload(query_id=self._query_id, query_log=query_log)
  
  # TODO: create the workload with corresponding frequency
  def all_with_frequency(self, frequency_data) -> Workload:
    """Generates a Workload of all parameters with corresponding frequency."""
    # get frequency  
    query_log = []
    for parameters in self._parameters:
      freq = frequency_data.get(json.dumps(parameters), 1)
      query_log.append(
          QueryInstance(execution_frequency=freq, parameters=parameters))
    
    # print(query_log)

    return Workload(query_id=self._query_id, query_log=query_log)

  def random_sample(self, n: int) -> Workload:
    """Generates a Workload of n random parameters without replacement."""
    sampled_parameters = random.sample(self._parameters, n)
    query_log = []
    for parameters in sampled_parameters:
      query_log.append(
          QueryInstance(execution_frequency=1, parameters=parameters))

    return Workload(query_id=self._query_id, query_log=query_log)


def create_query_batch(
    plan_ids: List[int],
    batch_workload: Workload) -> List[database_simulator.PlannedQuery]:
  """Creates a PlannedQuerys for the crossproduct of plan_ids and workload.

  This utility function generates a commonly expected request to be passed to
  DatabaseClient.execute_timed_batch().

  Args:
    plan_ids: The query plans to execute across all parameters.
    batch_workload: The workload of parameter bindings to execute with every
      plan.

  Returns:
    A list of PlannedQuerys delineating the crossproduct (query plan, parameter
    binding) combinations.
  """
  batch = []

  for query_instance, plan_id in itertools.product(batch_workload.query_log,
                                                   plan_ids):
    batch.append(
        database_simulator.PlannedQuery(
            query_id=batch_workload.query_id,
            plan_id=plan_id,
            parameters=query_instance.parameters))

  return batch


def shuffle(workload: Workload, seed: Optional[int] = None) -> None:
  """Shuffles the order of the queries in the query_log in place."""
  random.Random(seed).shuffle(workload.query_log)


def split(
    workload: Workload,
    first_half_count: Optional[int] = None,
    first_half_fraction: Optional[float] = None) -> Tuple[Workload, Workload]:
  """Splits the workload query log into two by counts or fractionage.

  The workload is split directly on the count of QueryInstance elements in the
  workload query log and does not consider QueryInstance execution_frequency.

  Exactly one of first_half_count and first_half_fraction must be provided.

  Args:
    workload: The workload to split.
    first_half_count: The number of queries to appear in the first workload
      after the split. If first_half_count is greater than the number of queries
      in workload, then all the queries in the workload will appear in the first
      half. The value must be >= 0.
    first_half_fraction: The fraction of queries to appear in the first workload
      after the split. The value must be in [0, 1].

  Returns:
    A tuple containing:
      1. A workload with first_half_count or first_half_fraction of the queries
        from the original workload.
      2. A workload containing the remaining queries, if any, from the original
        workload.

  Raises:
    ValueError: If both or neither of first_half_count and first_half_fraction
      are provided. Also raises if first_half_count is < 0 or
      first_half_fraction is
      not [0,1].
  """
  if ((first_half_count is None and first_half_fraction is None) or
      (first_half_count is not None and first_half_fraction is not None)):
    raise ValueError(
        "Exactly one of first_half_count and first_half_fraction must be provided."
    )
  if first_half_count is not None:
    if first_half_count < 0:
      raise ValueError(
          f"The first_half_count must be >= 0, but is {first_half_count}")

  else:
    if first_half_fraction < 0.0 or first_half_fraction > 1.0:
      raise ValueError(
          f"The first_half_fraction must be [0, 1], but is {first_half_fraction}"
      )
    first_half_count = int(first_half_fraction * len(workload.query_log))

  return (Workload(
      query_id=workload.query_id,
      query_log=workload.query_log[:first_half_count]),
          Workload(
              query_id=workload.query_id,
              query_log=workload.query_log[first_half_count:]))
