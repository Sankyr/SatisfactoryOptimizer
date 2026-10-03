import json
import math
from dataclasses import dataclass, field
from typing import Union, Any

from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import numpy as np

import itertools
import satisfactory_parser

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

@dataclass
class Vector2:
  x: float
  y: float

  def distance(self, otherVector2):
    return math.sqrt(math.pow(self.x - otherVector2.x, 2) + math.pow(self.y - otherVector2.y, 2))

@dataclass
class FactoryNode:
  id: int
  recipeName: str
  inputRate: dict[str, float]
  outputRate: dict[str, float]
  inputs: list[tuple[int, str, str, float]]
  outputs: list[tuple[int, str, str, float]]
  position: Vector2 = field(default_factory=lambda: Vector2(0, 0))

  def __str__(self):
    inputs = ', '.join([f'{{"id": {v[0]}, "recipe": "{v[1]}", "item": "{v[2]}", "rate": {v[3]}}}' for v in self.inputs])
    outputs = ', '.join([f'{{"id": {v[0]}, "recipe": "{v[1]}", "item": "{v[2]}", "rate": {v[3]}}}' for v in self.outputs])
    return f'"{self.id}": {{"recipeName": "{self.recipeName}", "x": {self.position.x}, "y": {self.position.y}, "input": [{inputs}], "output": [{outputs}] }}'

def ReadResourceNodeData():
  # Open and read the JSON file
  with open('node_locations.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)

def ReadFactoryData():
  # Open and read the JSON file
  #with open('max_factory.json', 'r', encoding="utf8") as f:
  with open('true_max_factory.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)

@dataclass
class RecipeItem:
  recipeName: str
  ingredient: str
  rate: float
  adjacentRatio: float
  adjacentProductRatios: list[tuple[str, float]]  # (product, ratio)

def CreateRecipeItemList(max_factory) -> list[RecipeItem]:
  resource_names = set()
  for r in satisfactory_parser.RESOURCES:
    resource_names.update({f'{r.display_name} (Impure)', f'{r.display_name} (Normal)', f'{r.display_name} (Pure)'})
  recipe_item_list = []
  for recipe_name, recipe in max_factory.items():
    if recipe_name == '_COMMENTS' or recipe_name in resource_names:
      continue
    ingredientList = list(recipe['ingredients'].keys())
    if 'Desc_Water_C' in ingredientList:
      ingredientList.remove('Desc_Water_C')
    for i, ingredient in enumerate(ingredientList):
      if i < len(ingredientList) - 1:
        recipe_item_list.append(RecipeItem(recipe_name, ingredient, recipe['ingredients'][ingredient], adjacentRatio=recipe['ingredients'][ingredientList[i + 1]] / recipe['ingredients'][ingredient], adjacentProductRatios=[]))
      else:
        productsList = list(recipe['products'].keys())
        if 'Desc_Water_C' in productsList:
          productsList.remove('Desc_Water_C')
        recipe_item_list.append(RecipeItem(recipe_name, ingredient, recipe['ingredients'][ingredient], adjacentRatio=-1, adjacentProductRatios=[(productName, recipe['products'][productName] / recipe['ingredients'][ingredient]) for productName in productsList]))
  print(len(recipe_item_list))
  print(recipe_item_list)
  return recipe_item_list

@dataclass
class ProductInfo:
  product: str
  ingredient_index_ratio_pairs: list[tuple[int, float]]
  product_recipe_item_indices: list[int]

def GetProductInfo(recipe_item_list: list[RecipeItem]) -> dict[str, tuple[int, int, float]]:
  recipe_item_to_index = {f'{recipe_item.recipeName}:{recipe_item.ingredient}': i for i, recipe_item in enumerate(recipe_item_list)}
  item_to_used_in_list = {}
  for i, recipe_item in enumerate(recipe_item_list):
    if recipe_item.ingredient not in item_to_used_in_list:
      item_to_used_in_list[recipe_item.ingredient] = []
    item_to_used_in_list[recipe_item.ingredient].append(i)

  ingredient_ratio_pairs_per_product = {}
  for i, recipe_item in enumerate(recipe_item_list):
    for product, ratio in recipe_item.adjacentProductRatios:
      if product not in ingredient_ratio_pairs_per_product:
        ingredient_ratio_pairs_per_product[product] = []
      ingredient_ratio_pairs_per_product[product].append((i, ratio))
  print(recipe_item_to_index)
  print(f'item: [used in recipe]: {item_to_used_in_list}')
  print(f'item: [(made from recipe, at ratio)]: {ingredient_ratio_pairs_per_product}')
  return ingredient_ratio_pairs_per_product, item_to_used_in_list

def CreateResourceNodes(resource_node_data) -> tuple[list[FactoryNode], list[str], int]:
  node_id = 0
  resource_recipes = []
  resource_nodes = []
  for [node_type, nodes] in resource_node_data.items():
    resource_recipes.append(f'Resource: {node_type}')
    for node_data in nodes:
      resource_nodes.append(
        FactoryNode(node_id, resource_recipes[-1], {}, {node_type: node_data['rate']}, [], [], Vector2(node_data['x'], node_data['y'])))
      node_id += 1
  return resource_nodes, resource_recipes, node_id

def InitializeFactories(starting_id: int, factory_positions: list[Vector2]):
  factory_nodes = []
  for factory_position in factory_positions:
    factory_nodes.append(FactoryNode(starting_id, '', {}, {}, [], [], factory_position))
    starting_id += 1
  return factory_nodes

def CreateDistanceTensor(resource_nodes: list[FactoryNode], factories: list[FactoryNode], recipe_list: list[str]):
  tensor_shape = (len(resource_nodes) + len(factories), len(factories), len(recipe_list))
  print(f'Creating distance tensor of shape: {tensor_shape}, tensor size: {tensor_shape[0] * tensor_shape[1] * tensor_shape[2]}')
  distance_matrix = np.zeros((tensor_shape[0], tensor_shape[1]), dtype=float)
  for i in range(len(resource_nodes)):
    for j in range(len(factories)):
      distance_matrix[i][j] = resource_nodes[i].position.distance(factories[j].position)
  for i in range(len(factories)):
    for j in range(len(factories)):
      distance_matrix[i + len(resource_nodes)][j] = factories[i].position.distance(factories[j].position)

  distance_matrix = distance_matrix[:, :, np.newaxis]
  distance_matrix = np.tile(distance_matrix, (1, 1, tensor_shape[2]))
  return distance_matrix, tensor_shape

def GetTensorIndex(pos, tensor_shape):
  return pos[0] * tensor_shape[1] * tensor_shape[2] + pos[1] * tensor_shape[2] + pos[2]

def CreateConstraints(resource_nodes: list[FactoryNode], factories, recipe_items: list[RecipeItem], ingredient_ratio_pairs_per_product, item_to_indices, tensor_shape):
  row_indices = []
  col_indices = []
  data = []
  num_rows = 0

  b_eq = []

  ub_row_indices = []
  ub_col_indices = []
  ub_data = []
  num_ub_rows = 0

  b_ub = []
  # resource node rate constraint
  for i, resource_node in enumerate(resource_nodes):
    for j, recipe_item in enumerate(recipe_items):
      if list(resource_node.outputRate.keys())[0] == recipe_item.ingredient:
        row_indices.extend([num_rows] * tensor_shape[1])
        col_indices.extend([GetTensorIndex((i, f, j), tensor_shape) for f in range(tensor_shape[1])])
        data.extend([1] * tensor_shape[1])
    b_eq.append(list(resource_node.outputRate.values())[0])
    num_rows += 1
  #print(b_eq)
  # non-producing node constraint
  for i, resource_node in enumerate(resource_nodes):
    for j, recipe_item in enumerate(recipe_items):
      if list(resource_node.outputRate.keys())[0] != recipe_item.ingredient:
        row_indices.extend([num_rows] * tensor_shape[1])
        col_indices.extend([GetTensorIndex((i, f, j), tensor_shape) for f in range(tensor_shape[1])])
        data.extend([1] * tensor_shape[1])
    b_eq.append(0)
    num_rows += 1
  print(num_rows)
  print(f'row_indices: {len(row_indices)}, col_indices: {len(col_indices)}, data: {len(data)}, b_eq: {len(b_eq)}')
  # print('row_indices')
  # print(row_indices)
  # print('col_indices')
  # print(col_indices)
  # print('data')
  # print(data)
  # print('b_eq')
  # print(b_eq)

  # custom factory constraints
  for j, recipe_item in enumerate(recipe_items):
    if recipe_item.ingredient == 'Desc_OreUranium_C':
      row_indices.extend([num_rows] * tensor_shape[0])
      col_indices.extend([GetTensorIndex((f, 0, j), tensor_shape) for f in range(tensor_shape[0])])
      data.extend([1] * tensor_shape[0])
  b_eq.append(2100)
  num_rows += 1

  for j, recipe_item in enumerate(recipe_items):
    if recipe_item.ingredient == 'Desc_PlutoniumFuelRod_C':
      row_indices.extend([num_rows] * tensor_shape[0])
      col_indices.extend([GetTensorIndex((f, 0, j), tensor_shape) for f in range(tensor_shape[0])])
      data.extend([1] * tensor_shape[0])
  # b_eq.append(12.6)
  b_eq.append(7.875)
  num_rows += 1

  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.recipeName == 'Desc_SingularityCell_C Sink' and recipe_item.ingredient == 'Desc_SingularityCell_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(8)
  # num_rows += 1

  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_AlienPowerFuel_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(5)
  # num_rows += 1

  for j, recipe_item in enumerate(recipe_items):
    if recipe_item.recipeName == 'Ballistic Warp Drive (Slooped)' and recipe_item.ingredient == 'Desc_QuantumOscillator_C':
      row_indices.extend([num_rows] * tensor_shape[0])
      col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
      data.extend([1] * tensor_shape[0])
  b_eq.append(119.92688186957136)
  num_rows += 1

  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_SpaceElevatorPart_11_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(75)
  # num_rows += 1

  for j, recipe_item in enumerate(recipe_items):
    if recipe_item.ingredient == 'Desc_SpaceElevatorPart_12_C':
      row_indices.extend([num_rows] * tensor_shape[0])
      col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
      data.extend([1] * tensor_shape[0])
  #b_eq.append(320)
  b_eq.append(115.22965945470833)
  num_rows += 1

  for j, recipe_item in enumerate(recipe_items):
    if recipe_item.recipeName == 'AI Expansion Server (Slooped)' and recipe_item.ingredient == 'Desc_SpaceElevatorPart_6_C':
      row_indices.extend([num_rows] * tensor_shape[0])
      col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
      data.extend([1] * tensor_shape[0])
  b_eq.append(17.646236260857307)
  num_rows += 1

  for j, recipe_item in enumerate(recipe_items):
    if recipe_item.ingredient == 'Desc_SpaceElevatorPart_7_C':
      row_indices.extend([num_rows] * tensor_shape[0])
      col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
      data.extend([1] * tensor_shape[0])
  #b_eq.append(133.020559093718)
  b_eq.append(164.32840828987793)
  num_rows += 1
  # end custom factory constraints

  #TODO: create product -> ingredient inequality constraint
  itemsWithNoRecipe = set(
    ['Desc_QuantumEnergy_C', 'Desc_NitrogenGas_C', 'Desc_LiquidOil_C', 'Desc_OreIron_C', 'Desc_Stone_C',
     'Desc_OreCopper_C', 'Desc_Coal_C', 'Desc_OreGold_C', 'Desc_RawQuartz_C', 'Desc_OreUranium_C', 'Desc_SAM_C'])
  for i in range(len(factories)):
    for ingredient, recipe_item_indices in item_to_indices.items():
      if ingredient in itemsWithNoRecipe:
        continue
      for j in recipe_item_indices:
        row_indices.extend([num_rows] * (tensor_shape[1]))
        col_indices.extend([GetTensorIndex((i + len(resource_nodes), f, j), tensor_shape) for f in range(tensor_shape[1])])
        data.extend([-1] * tensor_shape[1])
      for j, ratio in ingredient_ratio_pairs_per_product[ingredient]:
        row_indices.extend([num_rows] * (tensor_shape[0]))
        col_indices.extend([GetTensorIndex((f, i, j), tensor_shape) for f in range(tensor_shape[0])])
        data.extend([ratio] * tensor_shape[0])
      b_eq.append(0)
      num_rows += 1

  # factory recipe constraint
  for i in range(len(factories)):
    for j, recipe_item in enumerate(recipe_items):
      # adjacent ratios are laid out in a chain with the last one set to -1 to signal the end
      # the last link in every chain should then connect to the product.
      # product ratios can only be an inequality because we don't know where the products are going in the ingredient matrix
      if recipe_item.adjacentRatio == -1:
        pass
        # # handle multiple products
        # for product_ratio in recipe_item.adjacentProductRatios:
        #   ub_row_indices.extend([num_ub_rows] * (tensor_shape[0]))
        #   ub_col_indices.extend([GetTensorIndex((f, i, j), tensor_shape) for f in range(tensor_shape[0])])
        #   ub_data.extend([product_ratio[1]] * tensor_shape[0])
        #   # glob all recipe_item variants of the product
        #   for product_index in item_to_indices[product_ratio[0]]:
        #     ub_row_indices.extend([num_ub_rows] * (tensor_shape[1]))
        #     ub_col_indices.extend([GetTensorIndex((i + len(resource_nodes), f, product_index), tensor_shape) for f in range(tensor_shape[1])])
        #     ub_data.extend([-1] * tensor_shape[1])
        #   b_ub.append(0)
        #   num_ub_rows += 1
      else:
        # ingredient pair ratio check across all production factories
        row_indices.extend([num_rows] * (2 * tensor_shape[0]))
        col_indices.extend(list(itertools.chain.from_iterable(
          [[GetTensorIndex((f, i, j), tensor_shape), GetTensorIndex((f, i, j + 1), tensor_shape)] for f in range(tensor_shape[0])])))
        data.extend([recipe_item.adjacentRatio, -1] * tensor_shape[0])
        b_eq.append(0)
        num_rows += 1

  print(num_rows)
  print(f'row_indices: {len(row_indices)}, col_indices: {len(col_indices)}, data: {len(data)}, b_eq: {len(b_eq)}')

  # total production constraint
  expected_resources = {'Desc_OreBauxite_C': -12300, 'Desc_Sulfur_C': -10800}
  for item, item_ratio_pairs in ingredient_ratio_pairs_per_product.items():
    # skip items with no ingredients
    if item in itemsWithNoRecipe:
      continue
    # ingredient pair ratio check across all production factories
    for item_ratio_pair in item_ratio_pairs:
      row_indices.extend([num_rows] * (tensor_shape[0] * tensor_shape[1]))
      col_indices.extend(list(itertools.chain.from_iterable(
        [[GetTensorIndex((j, k, item_ratio_pair[0]), tensor_shape) for j in range(tensor_shape[0])] for k in range(tensor_shape[1])])))
      data.extend([item_ratio_pair[1]] * (tensor_shape[0] * tensor_shape[1]))
    for product_index in item_to_indices[item]:
      row_indices.extend([num_rows] * (tensor_shape[0] * tensor_shape[1]))
      col_indices.extend(list(itertools.chain.from_iterable(
        [[GetTensorIndex((j, k, product_index), tensor_shape) for j in range(tensor_shape[0])] for k in range(tensor_shape[1])])))
      data.extend([-1] * (tensor_shape[0] * tensor_shape[1]))
    b_eq.append(expected_resources.get(item, 0))
    num_rows += 1

  # factory plan constraint
  for i, recipe_item in enumerate(recipe_items):
    row_indices.extend([num_rows] * (tensor_shape[0] * tensor_shape[1]))
    col_indices.extend(list(itertools.chain.from_iterable(
      [[GetTensorIndex((f0, f1, i), tensor_shape) for f1 in range(tensor_shape[1])] for f0 in range(tensor_shape[0])])))
    data.extend([1] * (tensor_shape[0] * tensor_shape[1]))
    b_eq.append(recipe_item.rate)
    num_rows += 1

  print(num_rows)
  print(f'row_indices: {len(row_indices)}, col_indices: {len(col_indices)}, data: {len(data)}, b_eq: {len(b_eq)}')

  row_indices = np.array(row_indices)
  col_indices = np.array(col_indices)
  data = np.array(data)

  A_ub_sparse = csr_matrix((ub_data, (ub_row_indices, ub_col_indices)),
                           shape=(num_ub_rows, tensor_shape[0] * tensor_shape[1] * tensor_shape[2]))
  b_ub = np.array(b_ub)
  A_eq_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(num_rows, tensor_shape[0] * tensor_shape[1] * tensor_shape[2]))
  b_eq = np.array(b_eq)
  return A_eq_sparse, b_eq
  #return A_ub_sparse, b_ub, A_eq_sparse, b_eq

def CreateFactoryNodesFromRoutingMatrix(x, factory_positions: list[Vector2], resource_nodes: list[FactoryNode], recipe_items: list[RecipeItem]):
  factory_nodes = []
  for i, resource_node in enumerate(resource_nodes):
    resource_factory = FactoryNode(i, resource_node.recipeName, {}, {}, [], [], resource_node.position)
    for j in range(len(factory_positions)):
      for k, recipe_item in enumerate(recipe_items):
        if x[i, j, k] != 0:
          resource_factory.outputs.append((j + len(resource_nodes), recipe_item.recipeName, recipe_item.ingredient, x[i, j, k]))
    factory_nodes.append(resource_factory)

  for i in range(len(factory_positions)):
    resource_factory = FactoryNode(i + len(resource_nodes), '', {}, {}, [], [], factory_positions[i])
    for j in range(len(resource_nodes)):
      for k, recipe_item in enumerate(recipe_items):
        if x[j, i, k] != 0:
          resource_factory.inputs.append((j, recipe_item.recipeName, recipe_item.ingredient, x[j, i, k]))
    for j in range(len(factory_positions)):
      for k, recipe_item in enumerate(recipe_items):
        if x[j + len(resource_nodes), i, k] != 0:
          resource_factory.inputs.append((j + len(resource_nodes), recipe_item.recipeName, recipe_item.ingredient, x[j + len(resource_nodes), i, k]))
    for j in range(len(factory_positions)):
      for k, recipe_item in enumerate(recipe_items):
        if x[i + len(resource_nodes), j, k] != 0:
          resource_factory.outputs.append((j + len(resource_nodes), recipe_item.recipeName, recipe_item.ingredient, x[i + len(resource_nodes), j, k]))
    factory_nodes.append(resource_factory)

  for factory_node in factory_nodes:
    print(f'{factory_node.id}, {factory_node.recipeName}')
    for item_input in factory_node.inputs:
      print(f'I: {item_input}')
    for output in factory_node.outputs:
      print(f'O: {output}')
  return factory_nodes

def GetRoute(resource_node_data):
  # load nodes
  # load factory plan
  # NOTE: recipe_items are based on the recipe products, resource nodes get a special recipe_item, sinked items get a special recipe_item
  # initialize factory location
  # create resource/factory x factory x recipe_item distance tensor
  # create resource rate constraint.
  #  - For each resource node of type r: sum of resource r in node slice (factory x recipe_item) = node rate
  # create resource blocking constraint.
  #  - For each resource node of type r:
  #    - sum of all recipe_item i where i =/= r in node slice (factory x recipe_item) = 0
  # create factory constraints.
  #  - For each factory:
  #    - For each recipe_item i:
  #      - For each recipe_item ingredient j: sum of i in factory slice (factory x recipe_item)
  #                                         - recipe weighted sum of j in factory slice (factory x _ x recipe_item) = 0
  #        recipe chain constraint (r0_1 * ingredient_0 - ingredient_1 = 0, r1_2 * ingredient_1 - ingredient_2 = 0, r2_p * ingredient_2 - product <= 0)
  #        sum r * ingredient - product = 0 for entire tensor
  #        set(product)
  #        product: [(ingredient index, product index, ingredient -> product ratio)]
  # create factory plan constraints.
  #  - For each recipe_item i: sum of i in factory slice (factory x factory) = factory plan rate
  # bounds: all allocations >= 0
  # minimize distance_tensor * x
  # info extraction: each element in x represents the amount of recipe_item i that goes from node/factory j to factory k
  resource_nodes, resource_recipes, next_id = CreateResourceNodes(resource_node_data)
  factory_positions = [
      # nuclear site
      Vector2(295296, 66740.7),  # 522
      Vector2(19518.5, -90037),

      # Vector2(59185.2, -8814.81),
      # Oil sites
      Vector2(-253111, 75555.6),
      Vector2(-40296.3, -6296.3),
      Vector2(47222.2, -2518.52),
      Vector2(178185, 205259),
      Vector2(260037, 112074),
      Vector2(169370, -93814.8),
      Vector2(144815, -209037),
      Vector2(-40925.9, -195815),
      Vector2(34629.6, -201481),

      # Nitrogen sites
      Vector2(-250593, -183852),
      Vector2(-146704, 120889),
      Vector2(84370.4, 105148),
      Vector2(125926, 256259),
      Vector2(214074, 135370),
      Vector2(223519, -311037),

      # Quartz sites
      # Vector2(-181333, -142296),
      # Vector2(-90666.7, 65481.5),
      # Vector2(34000, 118370),
      # Vector2(56666.7, 202111),
      # Vector2(56666.7, -132222),
      # Vector2(190778, -13851.9),
      # Vector2(306630, -299704),

      # Other sites
      Vector2(-268852, -45963),
      Vector2(-222259, -133481),
      Vector2(-144185, -31481.5),
      Vector2(-50370.4, 248074),
      Vector2(107667, 177556),
      Vector2(248074, 44703.7),
      Vector2(68000, -83740.7),
      Vector2(-43444.4, -142296),
      Vector2(-99481.5, -141037),
      Vector2(388481, -261926),
      Vector2(302852, -162444),
      Vector2(125296, -16370.4),
    # Vector2(0, 0),
    # Vector2(0, 0),
    # Vector2(0, 0),
    # Vector2(0, 0),


      # Vector2(-210926, -118370),
      # Vector2(-54148.1, 244926),
      # Vector2(298444, -161815),
      # Vector2(34629.6, -109556),
      # Vector2(-180704, 86888.9),
      # Vector2(119630, 204000),
      # Vector2(229815, 79963),
      # Vector2(91296.3, 39037),
      # Vector2(212815, -251852),
      # Vector2(-64222.2, -141667),
      # Vector2(-272000, -47222.2),
      # Vector2(-93185.2, 7555.56),
      # Vector2(207778, -28333.3),
    # Vector2(0, 0),

  ]
  # factory_positions = [
  #   Vector2(0, 0), # 522
  #   Vector2(304740, -176296),  # North East Desert
  #   Vector2(153629, 217222),  # South Oil Lake
  #   Vector2(261925, 53518),  # Swamp Water
  #   Vector2(27074, -210925),  # North Oil Coast
  #   Vector2(-39037, 244296),  # South
  #   Vector2(-44074, 50370),  # Red Forest
  #   Vector2(-246185, 76185),  # West Oil
  #   Vector2(-208407, -124037),  # North West
  #   Vector2(45333, -91296),  # North Starter
  # ]
  # factory_positions = [
  #   Vector2(-195185, -113333),  #
  #   Vector2(-278296, -155519),  #
  #   Vector2(-104519, -137889),  #
  #   Vector2(-271370, -50370.4),  #
  #   Vector2(-163074, -175037),  #
  #   Vector2(-149852, -28963),  #
  #   Vector2(-251222, 75555.6),  #
  #   Vector2(-146074, 115222),  #
  #   Vector2(-161815, 187630),  #
  #   Vector2(-39666.7, 41555.6),  #
  #   Vector2(-66740.7, 135370),  #
  #   Vector2(-42185.2, 240519),  #
  #   Vector2(5037.04, 156778),  #
  #   Vector2(152370, 217222),  #
  #   Vector2(55407.4, 199593),  #
  #   Vector2(221000, 147963),  #
  #   Vector2(269481, 110185),  #
  #   Vector2(101370, 159926),  #
  #   Vector2(85000, 103889),  #
  #   Vector2(260037, 55407.4),  #
  #   Vector2(125926, -18259.3),  #
  #   Vector2(71148.1, -83111.1),  #
  #   Vector2(-49740.7, -209667),  #
  #   Vector2(37148.1, -206519),  #
  #   Vector2(146704, -210296),  #
  #   Vector2(309778, -167481),  #
  #   Vector2(363296, -81851.9),  #
  #   Vector2(388481, -253111),  #
  #   Vector2(278926, -265704),  #
  #   Vector2(226037, -312296),  #
  #   Vector2(170630, -103259),  #
  #   Vector2(-41555.6, -140407),  #
  #   Vector2(46592.6, -5037.04),  #
  #   Vector2(-102630, 260667),  #
  #   # Vector2(0, 0),  #
  #   # Vector2(0, 0),  #
  #   # Vector2(0, 0),  #
  #   # Vector2(0, 0),  #
  # ]
  # factory_positions = [
  #   Vector2(-252481, -188259), # 522
  #   Vector2(-275778, -156148),  # North East Desert
  #   Vector2(-242407, -117111),  # South Oil Lake
  #   Vector2(-149222, -185741),  # Swamp Water
  #   Vector2(-182593, -139778),  # North Oil Coast
  #   Vector2(-183222, -113333),  # South
  #   Vector2(-100741, -139778),  # Red Forest
  #   Vector2(-270111, -42814.8),  # West Oil
  #   Vector2(-180074, -76185.2),  # North West
  #   Vector2(-162444, -30851.9),  # North Starter
  #   Vector2(-247444, 74925.9),  #
  #   Vector2(-108926, -47851.9),  #
  # ]
  factories = InitializeFactories(next_id, factory_positions)
  max_factory = ReadFactoryData()
  recipe_items = CreateRecipeItemList(max_factory) # resource_recipes + list(max_factory.keys())
  distances, tensor_shape = CreateDistanceTensor(resource_nodes, factories, recipe_items)
  # print(distances)
  ingredient_ratio_pairs_per_product, item_to_indices = GetProductInfo(recipe_items)
  # A_ub, b_ub, constraints, constraint_eq = CreateConstraints(resource_nodes, factories, recipe_items, ingredient_ratio_pairs_per_product, item_to_indices, tensor_shape)
  constraints, constraint_eq = CreateConstraints(resource_nodes, factories, recipe_items,
                                                             ingredient_ratio_pairs_per_product, item_to_indices,
                                                             tensor_shape)
  bounds = [(0, None)] * len(distances.flatten())

  # opt = linprog(c=distances.flatten(), A_ub=A_ub, b_ub=b_ub, A_eq=constraints, b_eq=constraint_eq, bounds=bounds, method='highs')
  opt = linprog(c=distances.flatten(), A_eq=constraints, b_eq=constraint_eq, bounds=bounds,
                method='highs')
  print(opt)
  # print(opt.x)
  print(len(opt.x))
  x = opt.x.reshape(tensor_shape)
  # print(x)
  # print(np.linspace(0, x.shape[0] - 1, x.shape[0]))
  # print(f'Uranium: {np.sum(x[0:(0 + 5), :, :])}')
  # print(f'Quartz: {np.sum(x[(0 + 5):(0 + 5 + 17), :, :])}')
  # print(f'Aluminum: {np.sum(x[(0 + 5 + 17):(0 + 5 + 17 + 17), :, :])}')
  # print(np.sum(x[140:(140+45), :, :]))
  # print(np.sum(x[:, :, 17]))
  # print(x[-1, :, :])
  return CreateFactoryNodesFromRoutingMatrix(x, factory_positions, resource_nodes, recipe_items)

def ReadSatisfactoryData():
  # Open and read the JSON file
  with open('satisfactory_data.json', 'r', encoding="utf-16") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)


def ParseItemDesc(itemClasses: list[dict[str, Any]]):
  items = {}
  stack_size_map = {'SS_ONE': 1, 'SS_SMALL': 50, 'SS_MEDIUM': 100, 'SS_BIG': 200, 'SS_HUGE': 500, 'SS_FLUID': 75}
  for item in itemClasses:
    items[item['ClassName']] = stack_size_map[item['mStackSize']] if 'mStackSize' in item else 0
  return items

ITEM_DESCRIPTOR_CLASSES = {
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGItemDescriptor\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGItemDescriptorNuclearFuel\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGItemDescriptorBiomass\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGItemDescriptorPowerBoosterFuel\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGResourceDescriptor\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGPowerShardDescriptor\'',
}

def ParseItems(satisfactoryJson: list[dict[str, Any]]):
  items = {}
  for data in satisfactoryJson:
    if data['NativeClass'] in ITEM_DESCRIPTOR_CLASSES:
      items.update(ParseItemDesc(data["Classes"]))
  return items

def main():
  resource_node_data = ReadResourceNodeData()
  factory_nodes = GetRoute(resource_node_data)
  print(f'{{{',\n'.join([str(f) for f in factory_nodes])}}}')
  # satisfactory_data = ReadSatisfactoryData()
  # stack_size_map = ParseItems(satisfactory_data)
  # total_distance = [0] * 20
  # total_stacks = [0] * 20
  # for factory_node in factory_nodes:
  #   for output in factory_node.outputs:
  #     dist = math.sqrt((factory_nodes[output[0]].position.x - factory_node.position.x) ** 2 + (factory_nodes[output[0]].position.y - factory_node.position.y) ** 2) / 100
  #     dist_bucket = int(dist // 500)
  #     total_distance[dist_bucket] += dist
  #     total_stacks[dist_bucket] += output[3] / stack_size_map[output[2]]
  #
  # dist_buckets = np.linspace(0, 10000, 21)
  # for i in range(0, 19):
  #   print(f'{dist_buckets[i]}-{dist_buckets[i+1]}: {total_distance[i]:.2f}, {total_stacks[i]:.2f}')

if __name__ == '__main__':
  main()