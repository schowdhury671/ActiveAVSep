import gzip
import json    


import os
from tqdm import tqdm
import pickle
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt



def load_points(points_file: str, transform=True, scene_dataset="replica"):
    r"""
    Helper method to load points data from files stored on disk and transform if necessary
    :param points_file: path to files containing points data
    :param transform: transform coordinate systems of loaded points for use in Habitat or not
    :param scene_dataset: name of scenes dataset ("replica", "mp3d", etc.)
    :return: points in transformed coordinate system for use with Habitat
    """
    points_data = np.loadtxt(points_file, delimiter="\t")
    if transform:
        if scene_dataset == "replica":
            points = list(zip(
                points_data[:, 1],
                points_data[:, 3] - 1.5528907,
                -points_data[:, 2])
            )
        elif scene_dataset == "mp3d":
            points = list(zip(
                points_data[:, 1],
                points_data[:, 3] - 1.5,
                -points_data[:, 2])
            )
        else:
            raise NotImplementedError
    else:
        points = list(zip(
            points_data[:, 1],
            points_data[:, 2],
            points_data[:, 3])
        )
    points_index = points_data[:, 0].astype(int)
    points_dict = dict(zip(points_index, points))
    assert list(points_index) == list(range(len(points)))
    return points_dict, points

def convert2str(point):
    return str(point[0]) + '_' + str(point[1]) + '_' + str(point[2])

def load_points_data(parent_folder, graph_file, transform=True, scene_dataset="replica"):
    r"""
    Main method to load points data from files stored on disk and transform if necessary
    :param parent_folder: parent folder containing files with points data
    :param graph_file: files containing connectivity of points per scene
    :param transform: transform coordinate systems of loaded points for use in Habitat or not
    :param scene_dataset: name of scenes dataset ("replica", "mp3d", etc.)
    :return: 1. points in transformed coordinate system for use with Habitat
             2. graph object containing information about the connectivity of points in a scene
    """
    points_file = os.path.join(parent_folder, 'points.txt')
    graph_file = os.path.join(parent_folder, graph_file)

    _, points = load_points(points_file, transform=transform, scene_dataset=scene_dataset)
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph

def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

class SubGraph_sampling():
  def __init__(self, points_to_sample=100, plotting=False):
        self.points_to_sample = points_to_sample
        self.plotting = False
        self.actions_dict = {'STOP':0, 'FORWARD':1, 'LEFT':2, 'RIGHT':3}

  def find_subgraphs(self, G, plotting=False):

      supported_nodes = {n for n, d in G.nodes(data="node_type") if d == "supported"}
      unsupported_nodes = {n for n, d in G.nodes(data="node_type") if d == "unsupported"}

      # print("supported nodes ", supported_nodes)

      # print("unsupported nodes ", unsupported_nodes)

      H = G.copy()
      H.remove_edges_from(
          (n, nbr, d)
          for n, nbrs in G.adj.items()
          if n in supported_nodes
          for nbr, d in nbrs.items()
          if nbr in unsupported_nodes
      )

      # print("H1 is ", H)

      H.remove_edges_from(
          (n, nbr, d)
          for n, nbrs in G.adj.items()
          if n in unsupported_nodes
          for nbr, d in nbrs.items()
          if nbr in supported_nodes
      )

      # print("H2 is ", H)

      G_minus_H = nx.DiGraph()
      G_minus_H.add_edges_from(set(G.edges) - set(H.edges))

      # print("G-H is ", G_minus_H)

      if plotting:
          _node_colors = [c for _, c in H.nodes(data="node_color")]
          _pos = nx.spring_layout(H)
          plt.figure(figsize=(8, 8))
          nx.draw_networkx_edges(H, _pos, alpha=0.3, edge_color="k")
          nx.draw_networkx_nodes(H, _pos, node_color=_node_colors)
          nx.draw_networkx_labels(H, _pos, font_size=14)
          plt.axis("off")
          plt.title("The stripped graph with the edges removed.")
          plt.show()
          _pos = nx.spring_layout(G_minus_H)
          plt.figure(figsize=(8, 8))
          ncl = [G.nodes[n]["node_color"] for n in G_minus_H.nodes]
          nx.draw_networkx_edges(G_minus_H, _pos, alpha=0.3, edge_color="k")
          nx.draw_networkx_nodes(G_minus_H, _pos, node_color=ncl)
          nx.draw_networkx_labels(G_minus_H, _pos, font_size=14)
          plt.axis("off")
          plt.title("The removed edges.")
          plt.show()

      subgraphs = [
          H.subgraph(c).copy() for c in nx.connected_components(H.to_undirected())
      ]

      return subgraphs, G_minus_H

  def locate_subgraph(self, start_node, subgraphs_of_G_ex):
    for _, sg in enumerate(subgraphs_of_G_ex):
      for node in sg.nodes():
        if start_node == node:
          return sg

    return None

  def sample_destination(self, start_node, node_to_point_dict, existing_destinations = dict()):
    node_arr = np.asarray(list(node_to_point_dict.keys()))
    dest_found = False
    attempts = 50
    i = 0
    while dest_found==False and i < attempts:
      perm_arr = node_arr[np.random.permutation(len(node_arr))]
      dest_node = perm_arr[0]
      if dest_node not in existing_destinations:
        if dest_node != start_node:
          dest_found = True
      i += 1

    if dest_found == False:
      dest_node = start_node

    return dest_node

  def find_shortest_path(self, s_graph, start_node, dest_node):
    return nx.shortest_path(s_graph,source=start_node,target=dest_node)

  def collect_actions(self, s_graph, path):
    start_node = path[0]
    actions = []
    corrected_path = [path[0]]
    for i in range(1,len(path)-1):
      prev = s_graph.nodes()[path[i-1]]['point']
      point_a = s_graph.nodes()[path[i]]['point']
      point_b = s_graph.nodes()[path[i+1]]['point']
      a_prev = np.array([point_a[0], point_a[2]]) - np.array([prev[0], prev[2]])
      b_a = np.array([point_b[0], point_b[2]]) - np.array([point_a[0], point_a[2]])
      angle = np.rad2deg(np.arcsin(np.cross(a_prev, b_a)/(np.sqrt(np.sum(a_prev**2))*np.sqrt(np.sum(b_a**2)))))
      if np.abs(angle) > 360:
        angle = angle % 360
      if angle >= 270:
        angle = angle - 360
      elif angle <= -270:
        angle = angle + 360
      if angle == 0:
        actions += [self.actions_dict['FORWARD']]
        corrected_path += [path[i]]
      if angle == 90:
        actions += [self.actions_dict['LEFT']]
        actions += [self.actions_dict['FORWARD']]
        corrected_path += [path[i-1]]
        corrected_path += [path[i]]
      elif angle == -90:
        actions += [self.actions_dict['RIGHT']]
        actions += [self.actions_dict['FORWARD']]
        corrected_path += [path[i-1]]
        corrected_path += [path[i]]
      elif angle == 180:
        actions += [self.actions_dict['RIGHT']]
        actions += [self.actions_dict['RIGHT']]
        actions += [self.actions_dict['FORWARD']]
        corrected_path += [path[i-1]]
        corrected_path += [path[i-1]]
        corrected_path += [path[i]]
      elif angle == -180:
        actions += [self.actions_dict['LEFT']]
        actions += [self.actions_dict['LEFT']]
        actions += [self.actions_dict['FORWARD']]
        corrected_path += [path[i-1]]
        corrected_path += [path[i-1]]
        corrected_path += [path[i]]
    actions += [self.actions_dict['STOP']]
    try:
      corrected_path += [path[i+1]]
    except:
      pass

    return actions, corrected_path

  def execute(self, graph, start_node, points_to_sample=None):
    if points_to_sample is None:
      points_to_sample = self.points_to_sample
    subgraphs_of_G_ex, _ = self.find_subgraphs(graph, plotting=self.plotting)
    s_graph = self.locate_subgraph(start_node, subgraphs_of_G_ex)
    existing_destinations = dict()

    node_to_point_dict = dict()
    for node in s_graph.nodes():
      node_to_point_dict[node] = s_graph.nodes()[node]['point']

    sampled_list = []

    for pts in range(points_to_sample):
      dest_node = self.sample_destination(start_node, node_to_point_dict, existing_destinations = existing_destinations)
      if dest_node != start_node:
        path = self.find_shortest_path(s_graph, start_node, dest_node)
        actions_list, corrected_path = self.collect_actions(s_graph, path)
      else:
        path = []
        # corrected_path = []
        actions_list = [self.actions_dict['STOP']]
      existing_destinations[dest_node] = 1
      sampled_list += [{'start':start_node, 'target':dest_node, 'graph':s_graph, 'path':corrected_path, 'actions_list':actions_list}]

    return sampled_list, node_to_point_dict


ROOT_DR = "/checkpoint/sagnikmjr2002/code/ActiveAVSepMovingSource/data"
assert os.path.isdir(ROOT_DR)

DATASETS_DR = f"{ROOT_DR}/active_datasets/v1"
assert os.path.isdir(DATASETS_DR)


SOURCE_DRN = "train_731243episodes/content"
DUMP_DRN = "train_movingSource_731243episodes/content"   # train_movingSource_731243episodes, train_movingSource_731243episodes_20episodesPerScene

# SOURCE_DRN = "val_100episodes"
# DUMP_DRN = "val_movingSource_100episodes"

# SOURCE_DRN = "valUnheard_100episodes"
# DUMP_DRN = "valUnheard_movingSource_100episodes"

# SOURCE_DRN = "test_1000episodes"
# DUMP_DRN = "test_movingSource_1000episodes"

# SOURCE_DRN = "testUnheard_1000episodes"
# DUMP_DRN = "testUnheard_movingSource_1000episodes"



####################################  first step is for modifying the scene json files with new key in the dict
# '''

# directory = "./content/"
directory = f"{DATASETS_DR}/{SOURCE_DRN}"

dump_directory = f"{DATASETS_DR}/{DUMP_DRN}"
if not os.path.isdir(dump_directory):
    os.makedirs(dump_directory)

for filename in tqdm(os.listdir(directory)[:]):    # :, 2
    file_path = os.path.join(directory, filename)
    
    if os.path.isdir(file_path):
        continue
    
    with gzip.open(file_path, "rb") as f:
      scene = json.loads(f.read(), encoding="utf-8")
      
      list_episodes = scene['episodes']
      
      for i in tqdm(range(len(list_episodes[:]))):   # :, 2, 20
       
        parent_folder = f'{ROOT_DR}/metadata/mp3d/' + scene['episodes'][0]['scene_id'].split("/")[0]
        points,graph = load_points_data(parent_folder, 'graph.pkl', scene_dataset="mp3d")
        
        start_point = list_episodes[i]['start_position']
        
        for node in graph.nodes():  
          temp_var = graph.nodes()[node]['point']
          
          start_point[0] = np.round(start_point[0],6)
          start_point[1] = np.round(start_point[1],6)
          start_point[2] = np.round(start_point[2],6)
          
                    
          #import pdb; pdb.set_trace()
          if np.round(temp_var[0],6) == start_point[0] and np.round(temp_var[1],6) == start_point[1] and np.round(temp_var[2],6) == start_point[2]:
            start_node = node
            break
        
        sampled_pts, node_to_point_dict = SubGraph_sampling(points_to_sample=1).execute(graph, start_node)
        # print("1: ", type(sampled_pts[0]['path'][-1]))
        list_episodes[i]['moving_source_positions'] = sampled_pts[0]['path']

      list_episodes_ = []
      for ele in list_episodes:
        if 'moving_source_positions' in ele:
          list_episodes_.append(ele)
      list_episodes = list_episodes_
    
    """ write to .gz file """
    """ Convert data to JSON """
    # json_str = json.dumps(list_episodes, default=str)
    json_str = json.dumps({'episodes': list_episodes}, default=str)

    """ Write JSON to .gz file """
    # dump_fp = file_path
    dump_fp = f"{dump_directory}/{file_path.split('/')[-1]}"
    print("writing file path ", dump_fp)

    with gzip.GzipFile(dump_fp, 'w') as fout:
      fout.write(json_str.encode('utf-8'))
        
# '''

""" Update on Sep 9, 2024: this block looks redundant """
##########################   second steps is for placing the list from that key to a different key
'''
# directory = "./content/"
directory = dump_directory

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    with gzip.open(file_path, "rb") as f:
      scene = json.loads(f.read()) #  , encoding="utf-8")
      
      list_episodes = scene # ['episodes']
      
      for i in range(len(list_episodes)):
        
        try:
          list_episodes[i]['moving_source_positions'][-1] = int(list_episodes[i]['moving_source_positions'][-1])
          print(file_path, i, 'str found')
        except:       
          pass
    
    # write to .gz file
    # Convert data to JSON
    json_str = json.dumps({'episodes': list_episodes})

    # Write JSON to .gz file
    print("writing file path ", file_path)
    with gzip.GzipFile(file_path, 'w') as fout:
      fout.write(json_str.encode('utf-8'))    
    
'''


##########################    dump to a new json  file with moving_source_scene
# '''

print("-" * 80)
print("1ST DUMP DONE")
print("-" * 80)

# directory = "./content/"
directory = dump_directory

total_episodeCount = 0
iterator_ = os.listdir(directory)
for filename in tqdm(iterator_):
    file_path = os.path.join(directory, filename)

    if os.path.isdir(file_path):
        continue
    
    # if 'moving' not in filename:
    assert 'moving' not in filename, print(filename)
    
    if True:
      with gzip.open(file_path, "rb") as f:
        scene = json.loads(f.read()) #  , encoding="utf-8")
        
        list_episodes = scene['episodes']
        # print('1: ', list_episodes[0].keys())
        l = []

        total_episodeCount += len(list_episodes)
        
        for ite in tqdm(range(len(list_episodes))): 
          
          # list_episodes[ite]['start_room'] = {'moving_source_positions':list_episodes[ite]['moving_source_positions']}
          # list_episodes[ite].pop('moving_source_positions')
          
          #import pdb; pdb.set_trace()
        
          # if 'moving_source_positions' not in list_episodes[ite]:
          #   print(ite, list_episodes[ite].keys())
          assert 'moving_source_positions' in list_episodes[ite]

          if len(list_episodes[ite]['moving_source_positions']) > 18:
            list_episodes[ite]['moving_source_positions'] = list_episodes[ite]['moving_source_positions'][:18]
                
          l += [{list_episodes[ite]['episode_id']:list_episodes[ite]['moving_source_positions']}] # ['start_room']
          list_episodes[ite]['start_room'] = None
      
      # write to .gz file
      # Convert data to JSON
      json_str = json.dumps({'episodes': list_episodes})
      
      json_str2 = json.dumps({'episodes': l})
  
      # Write JSON to .gz file
      print("writing file path ", file_path)
      with gzip.GzipFile(file_path, 'w') as fout:
        fout.write(json_str.encode('utf-8'))   
        
      # with gzip.GzipFile(file_path[:-8]+'_moving_source.json.gz', 'w') as fout:
      with gzip.GzipFile(file_path.split('.json.gz')[0] +'_moving_source.json.gz', 'w') as fout:
        fout.write(json_str2.encode('utf-8'))      

print(f"Total episode count: {total_episodeCount}")
         
# '''
