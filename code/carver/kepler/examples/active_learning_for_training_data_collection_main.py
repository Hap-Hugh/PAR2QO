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

"""Example script to reduce training data collection cost via active learning.

Training Kepler models with actual query executions can be expensive if a large
number of query instances need to be executed to build a model training data set
of query execution latencies per query plan. The Kepler training data collection
problem fits the general active learning paradigm where model training and
inference are cheap, while collecting additional training data examples (ie
query plan latencies for a query instance) can be quite expensive. We propose
active learning approaches may be used to make more deliberate decisions about
the query instances for which we should execute training data collection queries
on the database system.

This script is designed for illustrative purposes regarding how our collected
data set of actual query execution latencies combined with our database
simulation tools facilitate research into active learning approaches to reduce
the cost of Kepler. The active learning algorithm implemented is not
effective. It serves as a stand-in to show where one could experiment with and
evaluate active learning algorithm approaches. A researcher can measure and
compare the total query execution cost of various approaches without actually
incurring that significant cost.
"""

import collections
import copy
import json
import math
import os
from typing import Any, List, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import time

from kepler.data_management import database_simulator
from kepler.data_management import workload
from kepler.model_trainer import evaluation
from kepler.model_trainer import model_base
from kepler.model_trainer import sngp_multihead_model
from kepler.model_trainer import trainer
from kepler.model_trainer import trainer_util

_TRAINING_SPLIT_FRACTION = 0.8
_NUM_ACTIVE_LEARNING_SAMPLING_ITERATIONS = 10
# _NUM_INITIAL_QUERY_INSTANCES = 70 # MODIFY based on pool size (train distinct #)
# _NUM_NEXT_QUERY_INSTANCES_PER_ITERATION = 27 # MODIFY based on pool size
_NUM_INITIAL_QUERY_INSTANCES = flags.DEFINE_integer(
    "num_initial_query_instances",
    1,
    "Number of initial query instances to start with.",
)
_NUM_NEXT_QUERY_INSTANCES_PER_ITERATION = flags.DEFINE_integer(
    "num_next_query_instances_per_iteration",
    0,
    "Number of query instances to add per iteration.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")
_FREQUENCY_FILE = flags.DEFINE_string(
    "frequency_file", None,
    "Param list with corresponding frequency.")

# _CONFIDENCE_THRESHOLD = 0.9
_CONFIDENCE_THRESHOLD = flags.DEFINE_string(
    "confidence_threshold", None,
    "Confidence threshold for label prediction")
flags.mark_flag_as_required("confidence_threshold")

_PREDICTION_BATCH_SIZE = 1024

JSON = Any

_TESTING_PARAMETER_VALUES = flags.DEFINE_string(
    "testing_parameter_values",
    None,
    "File containing parameter values for testing set."
)
flags.mark_flag_as_required("testing_parameter_values")

_PLAN_HINTS_FILE = flags.DEFINE_string(
    "plan_hints_file",
    None,
    "File containing the plan hints."
)
flags.mark_flag_as_required("plan_hints_file")


_QUERY_METADATA = flags.DEFINE_string(
    "query_metadata",
    None,
    (
        "File containing query metadata which includes information on parameter"
        " values."
    ),
)
flags.mark_flag_as_required("query_metadata")

_EXECUTION_DATA = flags.DEFINE_string(
    "execution_data",
    None,
    "File containing query execution data across various parameter values.",
)
flags.mark_flag_as_required("execution_data")

_EXECUTION_METADATA = flags.DEFINE_string(
    "execution_metadata",
    None,
    "File containing query execution metadata across various parameter values.",
)
flags.mark_flag_as_required("execution_metadata")

_PREPROCESSING_CONFIG = flags.DEFINE_string(
    "preprocessing_config",
    None,
    "File containing preprocessing config information.",
)

_VOCAB_DATA_DIR = flags.DEFINE_string(
    "vocab_data_dir", None, "Folder containing distinct values for each column."
)
flags.mark_flag_as_required("vocab_data_dir")

_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")
_SEED = flags.DEFINE_integer(
    "seed", 0, "Seed used to shuffle workload before splitting."
)

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 20, "Number of epochs to train for."
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 64, "Training minibatch size.")
_VOCAB_SIZE_LIMIT = flags.DEFINE_integer(
    "vocab_size_limit", 200, "Maximum vocabulary size."
)


def _get_distinct_values(table, column):
  with open(
      os.path.join(_VOCAB_DATA_DIR.value, f"{table}-{column}-distinct_values")
  ) as f:
    return json.load(f)


def get_trained_model(
    database: database_simulator.DatabaseSimulator,
    client: database_simulator.DatabaseClient,
    metadata: JSON,
    preprocessing_config: JSON,
    plans: workload.KeplerPlanDiscoverer,
    workload_train: workload.Workload,
) -> model_base.ModelBase:
  """Returns a model trained on the provided workload."""

  # This example implementation re-"executes" the entire workload_train each
  # time for demo-building expediency and simplicity. The overhead is minimal
  # because we use to DatabaseSimulator to "execute" queries and this does not
  # affect the actual training of the model in any way. Technically, we're
  # simulating building a growing pool of training data and in a real system, we
  # wouldn't re-execute query instances. Indeed, our premise is that collecting
  # training data is expensive and we want to be very deliberate about which
  # query instances are executed.

  # The DatabaseSimulator's execution_count and execution_cost_ms attributes can
  # be used to measure the training data collection cost if they are cleared
  # before "executing" the training data collection queries each time.

  # Our specific modeling approach requires an execution time for every plan in
  # the plan set given a query instance. If an active learning modeling approach
  # can be even more lean such that it only requires *some* plans to be executed
  # per query instance, this can be implemented using the DatabaseSimlator and
  # DatabaseClient by building the PlannedQuery batch manually instead via
  # workload.create_query_batch.

  # Additionally fetch all default execution times to compute near-optimality.
  queries_train = workload.create_query_batch(plans.plan_ids, workload_train)
  queries_train_with_default = workload.create_query_batch(
      plans.plan_ids + [None], workload_train
  )
  query_execution_train_df = client.execute_timed_batch(
      planned_queries=queries_train
  )
  query_execution_train_with_default_df = client.execute_timed_batch(
      planned_queries=queries_train_with_default
  )

  default_latencies_train = evaluation.get_default_latencies(
      database=database, query_workload=workload_train
  )
  default_latencies_train = np.atleast_2d(default_latencies_train).T

  trainer_util.add_vocabulary_to_metadata(
      query_execution_train_df,
      metadata,
      _get_distinct_values,
      _VOCAB_SIZE_LIMIT.value,
  )

  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model_config = model_base.ModelConfig(
      layer_sizes=[64, 64, 64],
      dropout_rates=[0.0, 0.0, 0.0],
      learning_rate=3e-4,
      activation="relu",
      spectral_norm_multiplier=10.0,
      num_gp_random_features=128,
      loss=loss,
      metrics=[],
  )

  model = sngp_multihead_model.SNGPMultiheadModel(
      metadata, plans.plan_ids, model_config, preprocessing_config
  )

  t = trainer.NearOptimalClassificationTrainer(metadata, plans, model)
  x, y = t.construct_training_data(
      query_execution_train_with_default_df,
      default_relative=True,
      near_optimal_threshold=1.1,
  )

  # Construct sample weight matrix.
  all_latencies = np.array(query_execution_train_df["latency_ms"]).reshape(
      (-1, len(plans.plan_ids))
  )
  sample_weight = trainer_util.get_sample_weight(
      all_latencies, default_latencies_train
  )

  print("Training on %d samples" % len(y))
  t.train(
      x,
      y,
      epochs=_NUM_EPOCHS.value,
      batch_size=_BATCH_SIZE.value,
      sample_weight=sample_weight,
  )
  return model


class ActiveLearner:
  """Confidence-based query instance selection for training data."""

  def __init__(self, workload_parameter_pool: workload.Workload):
    self._parameter_pool = collections.deque(
        copy.deepcopy(workload_parameter_pool.query_log)
    )

  def propose_next_query_instances(
      self,
      model_predictor: Optional[
          sngp_multihead_model.SNGPMultiheadModelPredictor
      ],
      n: int,
  ) -> List[workload.QueryInstance]:
    """Proposes the next query instances for which to collect training data.

    The returned query instances are removed from the parameter pool.

    Args:
      model_predictor: A query plan predictor that also returns confidence
        values for each plan option. The None value is accepted for the
        bootstrap case when the model has not been trained on any data yet.
      n: The number of query instances to return.

    Returns:
      A list of QueryInstance for which to collect training data as recommended
      by this active learning algorithm.

      Specifically, if no model_predictor is provided, the first n query
      instances still remaining in the pool are returned. If a model_predictor
      is provided, the n query instances have the lowest confidence value for
      their max-confident plan. That is, the n query instances that are least
      confident about their top plan choice are returned.
    """

    if not model_predictor:
      query_instances = [self._parameter_pool[i] for i in range(n)]
      for query_instance in query_instances:
        del self._parameter_pool[0]
      return query_instances

    parameters = np.array(
        [query_instance.parameters for query_instance in self._parameter_pool]
    )

    # Run prediction in batches to avoid memory issues.
    confidences_list = []
    for i in range(math.ceil(len(parameters) / _PREDICTION_BATCH_SIZE)):
      parameters_list = parameters[
          i * _PREDICTION_BATCH_SIZE : (i + 1) * _PREDICTION_BATCH_SIZE
      ].T.tolist()
      _, auxiliary = model_predictor.predict(parameters_list)
      confidences_list.append(auxiliary["confidences"])
    confidences = np.concatenate(confidences_list)
    max_confidences = np.max(confidences, axis=1)
    max_confidence_tuples = [
        (i, max_confidence) for i, max_confidence in enumerate(max_confidences)
    ]
    query_instance_indices = sorted(max_confidence_tuples, key=lambda x: x[1])[
        :n
    ]
    # Remove the query instances with the lowest confidence predictions from the
    # parameter pool by index in reverse order so removals do not affect index
    # values of to-be-removed elements.
    query_instances = []
    for selected_query_instance in sorted(query_instance_indices, reverse=True):
      index = selected_query_instance[0]
      query_instances.append(self._parameter_pool[index])
      del self._parameter_pool[index]

    return query_instances


import csv
import numpy as np
from typing import List, Any

def save_predictions_to_csv(
    model_predictor, 
    workload_eval: workload.Workload, 
    output_file: str, 
    query_id: str,
    iteration: int,
    plan_hints_file: str
):
    """Save the predicted hint & its content into csv file."""
    # Step 0: Load plan hints from user input
    with open(plan_hints_file, 'r') as f:
        plan_hints_data = json.load(f)
    
    # Step 1: Construct model inputs and convert them to string
    eval_inputs = trainer_util.construct_multihead_model_inputs(workload_eval) # [[param for all query instance in table column]]
    eval_inputs_strings = [[str(value) for value in col] for col in np.array(eval_inputs).T]

    # Step 2: Get model predictions
    plan_selections, auxiliary = model_predictor.predict(eval_inputs)
    max_confidences = auxiliary["max_confidences"]
    print(f'plan_selections: {plan_selections} {max_confidences}')

    # Step 3: Write the results to a CSV file with iteration, params, and plan_id columns
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if iteration == 0:  # Write header only once
            writer.writerow(["iteration", "params", "plan_id", "confidence", "plan_content"])
            print("write header")

        for params_string, plan_id, confidence in zip(eval_inputs_strings, plan_selections, max_confidences):
            if plan_id is None:
                plan_id = -1
                plan_content = "No plan selected"
            else:
                # Retrieve the plan content from plan_hints_data using plan_id
                plan_content = plan_hints_data[query_id][plan_id]['hints'] if plan_id < len(plan_hints_data[query_id]) else "Invalid plan_id"
            
            writer.writerow([iteration, params_string, plan_id, confidence, plan_content])


def print_predictions(
    model_predictor, 
    workload_eval: workload.Workload, 
    output_file: str, 
    query_id: str,
    iteration: int,
    plan_hints_file: str
):
    """Save the predicted hint & its content into csv file."""
    # Step 1: Construct model inputs and convert them to string
    eval_inputs = trainer_util.construct_multihead_model_inputs(workload_eval) # [[param for all query instance in table column]]
    eval_inputs_strings = [[str(value) for value in col] for col in np.array(eval_inputs).T]

    # Step 2: Get model predictions
    plan_selections, auxiliary = model_predictor.predict(eval_inputs)
    max_confidences = auxiliary["max_confidences"]
    print(f'plan_selections: {plan_selections} {max_confidences}')

    # Step 3: Write the results to a CSV file with iteration, params, and plan_id columns
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
        
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if iteration == 0:  # Write header only once
            writer.writerow(["iteration", "params", "plan_id", "confidence"])

        for params_string, plan_id, confidence in zip(eval_inputs_strings, plan_selections, max_confidences):
            writer.writerow([iteration, params_string, plan_id, confidence])



def run_evaluation(
    model: model_base.ModelBase,
    metadata: JSON,
    plans: workload.KeplerPlanDiscoverer,
    workload_eval: workload.Workload,
    output_csv: str,
    query_id: str,
    iteration: int,
    plan_hints_file: str
):
  """Evaluates the current model against the evaluation workload."""

  model_predictor = sngp_multihead_model.SNGPMultiheadModelPredictor(
      model.get_model(),
      metadata,
      plan_cover=plans.plan_ids,
      confidence_threshold=float(_CONFIDENCE_THRESHOLD.value),
  )
  
  # Save predictions and plan content to CSV
  save_predictions_to_csv(model_predictor, workload_eval, output_csv, query_id, iteration, plan_hints_file)
#   print_predictions(model_predictor, workload_eval, "q1_0_1_predictions.csv", query_id, iteration, plan_hints_file)

#   eval_inputs = trainer_util.construct_multihead_model_inputs(workload_eval)
#   plan_selections, _ = model_predictor.predict(eval_inputs)

#   candidate_latencies = evaluation.get_candidate_latencies(
#       database=database,
#       query_workload=workload_eval,
#       plan_selections=plan_selections,
#   )
#   default_latencies = evaluation.get_default_latencies(
#       database=database, query_workload=workload_eval
#   )
#   optimal_latencies = evaluation.get_optimal_latencies(
#       client=client, query_workload=workload_eval, kepler_plan_discoverer=plans
#   )

#   # This example will log some results but does not otherwise print or
#   # incorporate the evaluation results into the active learning decision making
#   # or stopping condition.
#   evaluation.evaluate(
#       candidate_latencies=candidate_latencies,
#       default_latencies=default_latencies,
#       optimal_latencies=optimal_latencies,
#   )


def main(unused_argv):
  # CHANGE: calculate execution time
  start_time = time.time()
  
  query_id = _QUERY_ID.value
  with open(_QUERY_METADATA.value) as json_file:
    query_metadata = json.load(json_file)
    metadata = query_metadata[query_id]

  with open(_EXECUTION_DATA.value) as json_file:
    execution_data = json.load(json_file)

  with open(_EXECUTION_METADATA.value) as json_file:
    execution_metadata = json.load(json_file)

  if _PREPROCESSING_CONFIG.value:
    with open(_PREPROCESSING_CONFIG.value) as json_file:
      preprocessing_config = json.load(json_file)
  else:
    preprocessing_config = trainer_util.construct_preprocessing_config(metadata)

  plans = workload.KeplerPlanDiscoverer(
      query_execution_metadata=execution_metadata
  )
  database = database_simulator.DatabaseSimulator(
      query_execution_data=execution_data,
      query_execution_metadata=execution_metadata,
      estimator=database_simulator.LatencyEstimator.MIN,
  )
  client = database_simulator.DatabaseClient(database)
  workload_generator = workload.WorkloadGenerator(execution_data)
  
  # TODO: frequency implement
  if _FREQUENCY_FILE.value:
    with open(_FREQUENCY_FILE.value) as f:
        frequency_data = json.load(f)
    full_workload = workload_generator.all_with_frequency(frequency_data=frequency_data)
  else:
    full_workload = workload_generator.all()

  workload.shuffle(full_workload, seed=_SEED.value)

#   workload_parameter_pool, workload_eval = workload.split(
#       full_workload, first_half_fraction=_TRAINING_SPLIT_FRACTION
#   )

  # TODO: accept input testing files to test the trained model
  workload_parameter_pool = full_workload # take all training_distinct_data as training pool
  
  with open(_TESTING_PARAMETER_VALUES.value) as json_file:
    testing_set = json.load(json_file)
  
  # generate testing workloads
  testing_query_log = []
  
  for params in testing_set[query_id]["params"]:
      query_instance = workload.QueryInstance(execution_frequency=1, parameters=params)
      testing_query_log.append(query_instance)
      
  workload_eval = workload.Workload(query_id=query_id, query_log=testing_query_log)

  active_learner = ActiveLearner(workload_parameter_pool)

  # Bootstrap model training with initial set of query instances as training
  # examples.
  workload_train = workload.Workload(query_id=query_id, query_log=[])

  initial_query_instances = active_learner.propose_next_query_instances(
      model_predictor=None, n=_NUM_INITIAL_QUERY_INSTANCES.value
  ) # get most-unsure (low confidence) query instances for future model training
  workload_train.query_log.extend(initial_query_instances)

  model = get_trained_model(
      database=database,
      client=client,
      metadata=metadata,
      preprocessing_config=preprocessing_config,
      plans=plans,
      workload_train=workload_train,
  )

  # Initialize additional arguments - output_csv: str,
  output_csv = f'{_OUTPUT_DIR.value}/predictions/{query_id}_init_{_NUM_INITIAL_QUERY_INSTANCES.value}_batch_{_NUM_NEXT_QUERY_INSTANCES_PER_ITERATION.value}.csv'
  
  output_dir = os.path.dirname(output_csv)
  os.makedirs(output_dir, exist_ok=True)
  
  # Perform some basic model evaluation.
  run_evaluation(
      model=model,
      metadata=metadata,
      plans=plans,
      workload_eval=workload_eval,
      output_csv=output_csv,
      query_id=query_id,
      iteration=0,
      plan_hints_file=_PLAN_HINTS_FILE.value
  )

  # In this example, we select query instances to execute (here in simulation)
  # to obtain additional training data for a fixed number of iterations. This is
  # to keep the example sample. Other looping approaches could include a
  # while-loop condition on training or validation error.
  if _NUM_NEXT_QUERY_INSTANCES_PER_ITERATION.value != 0:
    for iteration in range(_NUM_ACTIVE_LEARNING_SAMPLING_ITERATIONS):
        model_predictor = sngp_multihead_model.SNGPMultiheadModelPredictor(
            model.get_model(),
            metadata,
            plan_cover=plans.plan_ids,
            confidence_threshold=float(_CONFIDENCE_THRESHOLD.value),
        )
        next_query_instances = active_learner.propose_next_query_instances(
            model_predictor, _NUM_NEXT_QUERY_INSTANCES_PER_ITERATION.value
        )
        workload_train.query_log.extend(next_query_instances)

        model = get_trained_model(
            database=database,
            client=client,
            metadata=metadata,
            preprocessing_config=preprocessing_config,
            plans=plans,
            workload_train=workload_train,
        )

        # Perform some basic model evaluation.
        run_evaluation(
        model=model,
        metadata=metadata,
        plans=plans,
        workload_eval=workload_eval,
        output_csv=output_csv,
        query_id=query_id,
        iteration=iteration+1, # start with 0
        plan_hints_file=_PLAN_HINTS_FILE.value
    )

  # CHANGE: end time
  end_time = time.time()
  execution_time = end_time - start_time
  
  # store logs
  metadata_file = os.path.join(_OUTPUT_DIR.value, "metadata.json")
  metadata = {
      "model_prediction_time": execution_time
  }
    
  with open(metadata_file, 'w') as f:
      json.dump(metadata, f, indent=4)

  logging.info(f"Script execution time: {execution_time} seconds")


if __name__ == "__main__":
  app.run(main)
