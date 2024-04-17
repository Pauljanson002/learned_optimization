# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities useful for training and meta-training."""
import os
from typing import Any, Sequence
import time

from absl import logging
from flax import serialization
import jax
from learned_optimization import filesystem as fs
from learned_optimization import profile
from learned_optimization import tree_utils
from learned_optimization.tasks import base as tasks_base

import numpy as onp
import jax.numpy as jnp


def save_state(path, state):
  fs.make_dirs(os.path.dirname(path))
  with fs.file_open(path, "wb") as fp:
    fp.write(serialization.to_bytes(state))


def load_state(path, state):
  logging.info("Restoring state %s", path)
  with fs.file_open(path, "rb") as fp:
    state_new = serialization.from_bytes(state, fp.read())
  tree = jax.tree_util.tree_structure(state)
  leaves = jax.tree_util.tree_leaves(state_new)
  return jax.tree_util.tree_unflatten(tree, leaves)



def timing_decorator(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to complete.")
        return result

    return wrapper


# @timing_decorator
def get_batches(task_family: tasks_base.TaskFamily,
                batch_shape: Sequence[int],
                split: str,
                numpy: bool = False) -> Any:
  """Get batches of data with the `batch_shape` leading dimension."""
  
  temp = next(task_family.datasets.split(split))
  # print(jax.tree_map(lambda x: x.shape,temp))
  # print(jax.tree_map(lambda x: x.device_buffers,temp))
  return temp
  return jax.tree_map(lambda x: jnp.reshape(x, batch_shape + (x.shape[0]//onp.prod(batch_shape),) + x.shape[1:],'C'),
                      temp)

def vec_get_batch(task_family, n_tasks, split, numpy=False):
   return next(task_family.datasets.split(split))


"""
# @timing_decorator
def get_batches(task_family: tasks_base.TaskFamily,
                batch_shape: Sequence[int],
                split: str,
                numpy: bool = False) -> Any:


  # print(batch_shape)
  # exit(0)
  if len(batch_shape) == 1:
    return vec_get_batch(task_family, batch_shape[0], numpy=numpy, split=split)
  
  elif len(batch_shape) == 2:
    datas_list = [
        vec_get_batch(task_family, batch_shape[1], numpy=numpy, split=split)
        for _ in range(batch_shape[0])
    ]
    if numpy:
      return tree_utils.tree_zip_onp(datas_list)
    else:
      return tree_utils.tree_zip_jnp(datas_list)
  elif len(batch_shape) == 3:
    datas_list = [
        get_batches(
            task_family, [batch_shape[1], batch_shape[2]],
            numpy=numpy,
            split=split) for _ in range(batch_shape[0])
    ]
    if numpy:
      return tree_utils.tree_zip_onp(datas_list)
    else:
      return tree_utils.tree_zip_jnp(datas_list)
  else:
    raise NotImplementedError()

import haiku as hk


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Elapsed time: {self.interval} seconds")

@profile.wrap()
# @timing_decorator
def vec_get_batch(task_family, n_tasks, split, numpy=False):
  # with jax.profiler.trace("/mnt/bnas/jax-trace", create_perfetto_link=True):

  start_time = time.time()
  if task_family.datasets is None:
    return ()
  to_zip = []
  for _ in range(n_tasks):
    to_zip.append(next(task_family.datasets.split(split)))
  end_time = time.time()

  print(f"for loop took {end_time - start_time} seconds to complete.")
  print('list',[type(x) for x in to_zip])
  print('list',jax.tree_map(lambda x: x.shape, to_zip))
  print('list',jax.tree_map(lambda x: type(x), to_zip))
  # exit(0)
  # _, tree_def = jax.tree_util.tree_flatten(to_zip[0])
  output_im = onp.zeros((len(to_zip),) + to_zip[0]['image'].shape)
  output_label = onp.zeros((len(to_zip),) + to_zip[0]['label'].shape)

  # print(output_im.shape,output_label.shape)
  # exit(0)


  def fast_cat(fm):
    with Timer() as t:
      i = [onp.expand_dims(x['image'],0) for x in fm]
      l = [onp.expand_dims(x['label'],0) for x in fm]
    return hk.data_structures.to_immutable_dict({'image': onp.concatenate(i, axis=0), 'label': onp.concatenate(l, axis=0)})
  
  def fast_cat_prealloc(fm,ip,lp):
    with Timer() as t:
      i = [onp.expand_dims(x['image'],0) for x in fm]
      l = [onp.expand_dims(x['label'],0) for x in fm]
      for x in range(len(i)):
        ip[x,...] = i[x]
        lp[x,...] = l[x]
    return hk.data_structures.to_immutable_dict({'image': ip, 'label': lp})


  start_time = time.time()
  # for i, data in enumerate(to_zip):
  #   jax.tree_map(lambda prealloc_array, single_data: prealloc_array.__setitem__(i, single_data), output, data)

  # output = tree_utils.tree_zip_onp(to_zip) if numpy else tree_utils.tree_zip_jnp(
  #     to_zip)
  # output = fast_cat(to_zip)
  output = fast_cat_prealloc(to_zip,output_im,output_label)
  end_time = time.time()

  print(f"output took {end_time - start_time} seconds to complete.")
  #  = preallocated_arrays
  print('output',jax.tree_map(lambda x: x.shape, output))

  # exit(0)
  return output

"""
# import time
# import jax
# import numpy as onp

# def vec_get_batch(task_family, n_tasks, split, numpy=False):
#     if task_family.datasets is None:
#         return ()
    
#     # Pre-fetch a sample to determine structure and size
#     sample_data = next(task_family.datasets.split(split))
#     _, tree_def = jax.tree_util.tree_flatten(sample_data)
    
#     # Pre-allocate memory based on the structure and size of the sample data
#     prealloc_shapes = jax.tree_map(lambda x: (n_tasks, *x.shape), sample_data)
#     preallocated_arrays = jax.tree_map(onp.empty, prealloc_shapes)
    
#     # Re-initialize the data list starting with the sample data
#     to_zip = [sample_data] + [next(task_family.datasets.split(split)) for _ in range(1, n_tasks)]

#     start_time = time.time()

#     # Manually fill in the preallocated arrays
#     for i, data in enumerate(to_zip):
#         jax.tree_map(lambda prealloc_array, single_data: prealloc_array.__setitem__(i, single_data), preallocated_arrays, data)

#     end_time = time.time()
#     print(f"Data loading and processing took {end_time - start_time} seconds to complete.")
#     print('output', jax.tree_map(lambda x: x.shape, preallocated_arrays))

#     return preallocated_arrays
