"""
======================= START OF LICENSE NOTICE =======================
  Copyright (C) 2024 JasmineXie. All Rights Reserved

  NO WARRANTY. THE PRODUCT IS PROVIDED BY DEVELOPER "AS IS" AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DEVELOPER BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE PRODUCT, EVEN
  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
======================== END OF LICENSE NOTICE ========================
  Primary Author: JasmineXie

"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))

from distribute_monitor.logger import LOGGER
from distribute_monitor.node import Node

  
CONF_NODES = {}

def load_conf(conf_file = 'nodes.json'):
  conf_filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), conf_file)
  file = None
  try:
    file = open(conf_filepath, 'r')
    conf = json.load(file)
    nodes = conf['nodes']
    assert nodes is not None
    nodes_len = len(nodes)
    if not (nodes_len&1):
      LOGGER.warning('majority might be multiple for {} nodes'.format(nodes_len))
    for node in nodes:
      node_info = nodes[node]
      CONF_NODES[node] = (node_info['ip_addr'], node_info['port'])
  finally:
    if file is not None:
      file.close()
      
Nodes = []
def get_all_nodes_status():
    results = []
    for node in Nodes:
        results.append(node.get_status())
    return results


load_conf()
for node in CONF_NODES:
  Nodes.append(Node(CONF_NODES[node][0], CONF_NODES[node][1]))

