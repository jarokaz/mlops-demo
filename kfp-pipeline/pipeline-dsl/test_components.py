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


from typing import NamedTuple
def retrieve_best_run(alpha:float, max_iter:int)->NamedTuple('Outputs',
                                                   [('alpha', float),
                                                    ('max_iter', int),
                                                    ('mlpipeline_ui_metadata', 'UI_metadata')]):

    import json

    markdown = (
        '**Alpha:**      {alpha}  \n'
        '**Max Iter:**   {max_iter}  \n'
    ).format(alpha=round(alpha, 5), max_iter=max_iter)
    
    metadata = {
        'outputs': [
            {
                'type': 'markdown',
                'storage': 'inline',
                'source': markdown
            }
        ]
    }

    return (alpha, max_iter, json.dumps(metadata))


