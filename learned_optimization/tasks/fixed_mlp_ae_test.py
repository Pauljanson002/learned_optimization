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

"""Tests for learned_optimizers.tasks.fixed_mlp."""

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.tasks import fixed_mlp_ae
from learned_optimization.tasks import test_utils

tasks = [
    "FixedMLPAE_cifar10_32x32x32_bs128",
    "FixedMLPAE_cifar10_256x256x256_bs128",
    "FixedMLPAE_cifar10_256x256x256_bs1024",
    "FixedMLPAE_cifar10_128x32x128_bs256",
    "FixedMLPAE_mnist_128x32x128_bs128",
    "FixedMLPAE_fashion_mnist_128x32x128_bs128",
]


class FixedMLPAETest(parameterized.TestCase):

  @parameterized.parameters(tasks)
  def test_tasks(self, task_name):
    task = getattr(fixed_mlp_ae, task_name)()
    test_utils.smoketest_task(task)


if __name__ == "__main__":
  absltest.main()