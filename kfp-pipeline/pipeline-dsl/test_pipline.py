# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import kfp
import os
import uuid
import time
import tempfile


from google.cloud import bigquery
from jinja2 import Template
from kfp.components import func_to_container_op
from kfp.gcp import use_gcp_secret
from kfp.dsl.types import GCPProjectID, GCSPath, GCRPath, GCPRegion, Integer, String, Float, List, Dict
from typing import NamedTuple

from test_components import retrieve_best_run
    

BASE_IMAGE = os.getenv("BASE_IMAGE")

retrieve_best_run_op = func_to_container_op(retrieve_best_run, base_image=BASE_IMAGE)



@kfp.dsl.pipeline(
    name='Test pipeline',
    description='Test pipeline'
)
def test_pipeline(
    alpha:float =0.0005,
    max_iter:int =100
    ):
    

    # Retrieve the best trial
    get_best_trial = retrieve_best_run_op(alpha, max_iter)
    

    
    kfp.dsl.get_pipeline_conf().add_op_transformer(use_gcp_secret('user-gcp-sa'))
    
