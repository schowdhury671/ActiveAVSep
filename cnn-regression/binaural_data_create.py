import gzip
import json    


import os
import pickle
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm



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
        self.filedir = '/fs/nexus-projects/ego_data/active_avsep/active-AV-dynamic-separation/data/audio_data/libriSpeech100Classes_MITMusic_ESC50/1s_chunks/train_preprocessed'
        self.all_sounds = os.listdir(self.filedir)

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

  def execute(self, graph, start_node, scene_name, mono_name, points_to_sample=None,):
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
        delta_x = node_to_point_dict[dest_node][0] - node_to_point_dict[start_node][0]
        delta_y = node_to_point_dict[dest_node][2] - node_to_point_dict[start_node][2]
      else:
        delta_x = node_to_point_dict[start_node][0] - node_to_point_dict[start_node][0]
        delta_y = node_to_point_dict[start_node][2] - node_to_point_dict[start_node][2]
      existing_destinations[dest_node] = 1

      
      # fils = [fil for fil in self.list_monos if mono_name in fil]
      fils = self.all_sounds
      cnt = len(fils)
      mono_name_chunk = fils[np.random.permutation(cnt)[0]]
      mono_file_path = filedir + '/' + mono_name_chunk
      assert os.path.isfile(mono_file_path)

      for az in [0, 90, 180, 270]:
        if az == 0:
           delta_x_az, delta_y_az = delta_x, delta_y
        elif az == 90:
           delta_x_az, delta_y_az = -delta_y, delta_x
        elif az == 180:
           delta_x_az, delta_y_az = -delta_x, -delta_y
        else:
           delta_x_az, delta_y_az = delta_y, -delta_x
        sampled_list += [{'binaural_rir_filename':'/fs/nexus-projects/ego_data/active_avsep/active-AV-dynamic-separation/data/binaural_rirs/mp3d/' + scene_name + '/' + str(az)+'/'+str(start_node)+'_'+str(dest_node)+'.wav', 'target': [delta_x_az, delta_y_az], 'mono_filename': mono_file_path}]

    return sampled_list



directory = "./content/"

list_all_wavs = []

for ite, filename in enumerate(tqdm(os.listdir(directory))):
    # if ite == 2:
    #    break
    file_path = os.path.join(directory, filename)
    if 'moving_source' not in file_path:
        with gzip.open(file_path, "rb") as f:
            scene = json.loads(f.read(), encoding="utf-8")
        
            list_episodes = scene['episodes']
        
            for i in range(len(list_episodes)):
        
                # print("@#@#@#@#@#@## scene episodes1 ", scene['episodes'][0])
                # print("******************* scene episodes2 ", scene['episodes'][0]['scene_id'])
                parent_folder = '/fs/nexus-projects/ego_data/active_avsep/sound-spaces/data/metadata/mp3d/' + scene['episodes'][0]['scene_id'].split("/")[0]
                
                points,graph = load_points_data(parent_folder, 'graph.pkl', scene_dataset="mp3d")
            
                start_point = list_episodes[i]['start_position']
            
                for node in graph.nodes():  
                    temp_var = graph.nodes()[node]['point']
            
                    if np.round(temp_var[0],6) == start_point[0] and np.round(temp_var[1],6) == start_point[1] and np.round(temp_var[2],6) == start_point[2]:
                        start_node = node
                        break
            
                list_all_wavs += SubGraph_sampling(points_to_sample=1).execute(graph, start_node, scene['episodes'][0]['scene_id'].split("/")[0], list_episodes[i]['info'][0]['sound'])

dict_wavs = {'train':list_all_wavs}

# import pdb; pdb.set_trace()
with open("train_wavs.json", "w") as outfile:
    # json_data refers to the above JSON
    json.dump(dict_wavs, outfile)
        
    
    
