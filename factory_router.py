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

RESOURCE_ITEMS = {'Desc_OreIron_C', 'Desc_Coal_C', 'Desc_Sulfur_C', 'Desc_SAM_C', 'Desc_OreBauxite_C', 'Desc_OreGold_C', 'Desc_OreCopper_C', 'Desc_Stone_C', 'Desc_RawQuartz_C', 'Desc_OreUranium_C', 'Desc_LiquidOil_C', 'Desc_NitrogenGas_C'}

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

@dataclass
class PlanningStructures:
  items: list[str]
  produced_by_per_item: dict[str, list[str]]
  consumed_by_per_item: dict[str, list[str]]
  resource_edges: list[tuple[str, str, str]]
  factory_edges: list[tuple[str, str, str]]
  producing_recipe_links: dict[str, tuple[str, float]]
  consuming_recipe_links: dict[str, tuple[str, float]]
  indices_per_consuming_factory_recipe_item: dict[str, list[int]] = field(default_factory=dict)
  indices_per_produced_recipe_item: dict[str, list[int]] = field(default_factory=dict)
  indices_per_recipe_product: dict[str, list[int]] = field(default_factory=dict)
  indices_per_resource_node: dict[int, list[int]] = field(default_factory=dict)

@dataclass
class ItemEdge:
  item: str

  producing_factory: int
  producing_recipe: str

  consuming_factory: int
  consuming_recipe: str

  rate: float = 0

  def get_distance(self, resource_nodes: list[FactoryNode], factory_nodes: list[Vector2]) -> float:
    if self.producing_factory < len(resource_nodes):
      producing_factory_loc = resource_nodes[self.producing_factory].position
    else:
      producing_factory_loc = factory_nodes[self.producing_factory - len(resource_nodes)]
    return producing_factory_loc.distance(factory_nodes[self.consuming_factory])

  def encode(self, next_row_index, planning_structures: PlanningStructures, num_resource_nodes: int):
    row_indices = []
    col_indices = []
    data = []
    b_eq = []
    # broken
    if f'{self.producing_recipe}:{self.item}' in planning_structures.producing_recipe_links:
      child_item, child_rate = planning_structures.producing_recipe_links[f'{self.producing_recipe}:{self.item}']
      # producing_factory = all factories, producing_recipe_edge = child_item, consuming_factory = this.producing_factory, consuming_recipe = this.producing_recipe
      # child constraint
      consuming_factory_recipe_item = f'{self.producing_factory - num_resource_nodes}:{self.producing_recipe}:{child_item}'
      if consuming_factory_recipe_item in planning_structures.indices_per_consuming_factory_recipe_item:
        indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
        col_indices.extend(indices)
        row_indices.extend([next_row_index] * len(indices))
        data.extend([child_rate] * len(indices))
        indices = planning_structures.indices_per_produced_recipe_item[f'{self.producing_factory}:{self.producing_recipe}:{self.consuming_factory}:{self.consuming_recipe}:{self.item}']
        col_indices.extend(indices)
        row_indices.extend([next_row_index] * len(indices))
        data.extend([-1] * len(indices))
        b_eq.append(0)
        next_row_index += 1

    # if f'{self.consuming_recipe}:{self.item}' in planning_structures.consuming_recipe_links:
    #   sibling_item, sibling_rate = planning_structures.consuming_recipe_links[f'{self.consuming_recipe}:{self.item}']
    #   # producing_factory = all factories, producing_recipe = sibling_item, consuming_factory = this.consuming_factory, consuming_recipe = this.consuming_recipe
    #   # sibling constraint
    #   indices = planning_structures.indices_per_consuming_factory_recipe_item[f'{self.consuming_factory}:{self.consuming_recipe}:{sibling_item}']
    #   col_indices.extend(indices)
    #   row_indices.extend([next_row_index] * len(indices))
    #   data.extend([sibling_rate] * len(indices))
    #   indices = planning_structures.indices_per_produced_recipe_item[f'{self.producing_factory}:{self.producing_recipe}:{self.consuming_factory}:{self.consuming_recipe}:{self.item}']
    #   col_indices.extend(indices)
    #   row_indices.extend([next_row_index] * len(indices))
    #   data.extend([-1] * len(indices))
    #   b_eq.append(0)
    #   next_row_index += 1

    return row_indices, col_indices, data, b_eq, next_row_index


@dataclass
class RecipeItem:
  recipeName: str
  ingredient: str
  rate: float
  adjacentRatio: float
  adjacentProductRatios: list[tuple[str, float]]  # (product, ratio)

def ReadResourceNodeData():
  # Open and read the JSON file
  with open('node_locations.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)

def ReadFactoryData():
  # Open and read the JSON file
  #with open('max_factory.json', 'r', encoding="utf8") as f:
  # with open('true_max_factory.json', 'r', encoding="utf8") as f:
  with open('near_max_factory_with_extras.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)


def get_planning_structures(max_factory) -> PlanningStructures:
  products = []
  ingredients = []
  produced_by_per_item = {}
  consumed_by_per_item = {}
  producing_recipe_links: dict[str, tuple[str, float]] = {}
  consuming_recipe_links: dict[str, tuple[str, float]] = {}
  for recipe_name, recipe in max_factory.items():
    if recipe_name == '_COMMENTS':
      continue

    products.extend(list(recipe['products'].keys()))
    for product, product_rate in recipe['products'].items():
      if product not in produced_by_per_item:
        produced_by_per_item[product] = []
      produced_by_per_item[product].append(recipe_name)
      recipe_product = f'{recipe_name}:{product}'
      if recipe_product not in producing_recipe_links:
        if not recipe['ingredients']:
          continue
        first_ingredient = list(recipe['ingredients'].keys())[0]
        producing_recipe_links[recipe_product] = (first_ingredient, product_rate / recipe['ingredients'][first_ingredient])

    ingredients_for_recipe = list(recipe['ingredients'].keys())
    ingredients.extend(ingredients_for_recipe)
    for i, ingredient in enumerate(ingredients_for_recipe):
      if ingredient not in consumed_by_per_item:
        consumed_by_per_item[ingredient] = []
      consumed_by_per_item[ingredient].append(recipe_name)
      recipe_ingredient = f'{recipe_name}:{ingredient}'
      if recipe_ingredient not in consuming_recipe_links:
        if i >= len(ingredients_for_recipe) - 1:
          continue
        next_ingredient_rate = ingredients_for_recipe[i + 1]
        consuming_recipe_links[recipe_ingredient] = (next_ingredient_rate,
                                                  recipe['ingredients'][ingredient] / recipe['ingredients'][next_ingredient_rate])

  products = list(dict.fromkeys(products))
  ingredients = list(dict.fromkeys(ingredients))
  items = list(dict.fromkeys(products + ingredients))

  resource_edges = []
  factory_edges = []

  resources = {'Desc_OreIron_C', 'Desc_Coal_C', 'Desc_Sulfur_C', 'Desc_SAM_C', 'Desc_OreBauxite_C', 'Desc_OreGold_C', 'Desc_OreCopper_C', 'Desc_Stone_C', 'Desc_RawQuartz_C', 'Desc_OreUranium_C', 'Desc_LiquidOil_C', 'Desc_NitrogenGas_C'}
  non_natural_resource = {'Sulfur (Iron)', 'Coal (Limestone)', 'Bauxite (Copper)', 'Bauxite (Caterium)', 'Nitrogen Gas (Caterium)'}



  for item in items:
    for produced_by in produced_by_per_item[item]:
      for consumed_by in consumed_by_per_item[item]:
        if item in resources and produced_by not in non_natural_resource:
          resource_edges.append((item, '', consumed_by))
        if item not in resources or (item in resources and produced_by in non_natural_resource):
          factory_edges.append((item, produced_by, consumed_by))

  resource_edges = list(dict.fromkeys(resource_edges))
  print(len(items))
  print(items)
  print(produced_by_per_item)
  print(consumed_by_per_item)
  print(len(resource_edges))
  print(resource_edges)
  print(len(factory_edges))
  print(factory_edges)
  print(len(producing_recipe_links))
  print(producing_recipe_links)
  print(len(consuming_recipe_links))
  print(consuming_recipe_links)
  planning_structures = PlanningStructures(
    items=items,
    produced_by_per_item=produced_by_per_item,
    consumed_by_per_item=consumed_by_per_item,
    resource_edges=resource_edges,
    factory_edges=factory_edges,
    producing_recipe_links=producing_recipe_links,
    consuming_recipe_links=consuming_recipe_links
  )
  return planning_structures

def CreateRecipeItemList(max_factory) -> list[RecipeItem]:
  resource_names = set()
  for r in satisfactory_parser.RESOURCE_NODES:
    resource_names.update({f'{r.display_name} (Impure)', f'{r.display_name} (Normal)', f'{r.display_name} (Pure)'})
  for r in satisfactory_parser.RESOURCE_WELLS:
    resource_names.add(r.display_name)
  recipe_item_list = []
  for recipe_name, recipe in max_factory.items():
    if recipe_name == '_COMMENTS' or recipe_name in resource_names:
      continue
    ingredientList = list(recipe['ingredients'].keys())
    # if 'Desc_Water_C' in ingredientList:
    #   ingredientList.remove('Desc_Water_C')
    for i, ingredient in enumerate(ingredientList):
      if i < len(ingredientList) - 1:
        recipe_item_list.append(RecipeItem(recipe_name, ingredient, recipe['ingredients'][ingredient], adjacentRatio=recipe['ingredients'][ingredientList[i + 1]] / recipe['ingredients'][ingredient], adjacentProductRatios=[]))
      else:
        productsList = list(recipe['products'].keys())
        # if 'Desc_Water_C' in productsList:
        #   productsList.remove('Desc_Water_C')
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

def CreateConstraints(resource_nodes: list[FactoryNode], factories, recipe_items: list[RecipeItem], ingredient_ratio_pairs_per_product, item_to_indices, tensor_shape, max_factory):
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
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_OreUranium_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 0, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(2100)
  # num_rows += 1
  #
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_PlutoniumFuelRod_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 0, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['Plutonium Fuel Rod (Sink)']['ingredients']['Desc_PlutoniumFuelRod_C'])
  # num_rows += 1
  #
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.recipeName == 'Singularity Cell (Portal)' and recipe_item.ingredient == 'Desc_SingularityCell_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['Singularity Cell (Portal)']['ingredients']['Desc_SingularityCell_C'])
  # num_rows += 1
  #
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_AlienPowerFuel_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['Alien Power Matrix (Power Augmenter)']['ingredients']['Desc_AlienPowerFuel_C'])
  # num_rows += 1
  #
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.recipeName == 'Ballistic Warp Drive (Slooped)' and recipe_item.ingredient == 'Desc_QuantumOscillator_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['Ballistic Warp Drive (Slooped)']['ingredients']['Desc_QuantumOscillator_C'])
  # num_rows += 1
  #
  # # for j, recipe_item in enumerate(recipe_items):
  # #   if recipe_item.ingredient == 'Desc_SpaceElevatorPart_11_C':
  # #     row_indices.extend([num_rows] * tensor_shape[0])
  # #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  # #     data.extend([1] * tensor_shape[0])
  # # b_eq.append(75)
  # # num_rows += 1
  #
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_SpaceElevatorPart_12_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['AI Expansion Server (Sink)']['ingredients']['Desc_SpaceElevatorPart_12_C'])
  # num_rows += 1

  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.recipeName == 'AI Expansion Server (Slooped)' and recipe_item.ingredient == 'Desc_SpaceElevatorPart_6_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['Ballistic Warp Drive (Slooped)']['ingredients']['Desc_QuantumOscillator_C'])
  # num_rows += 1

  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_SpaceElevatorPart_7_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # #b_eq.append(133.020559093718)
  # b_eq.append(max_factory['Ballistic Warp Drive (Slooped)']['ingredients']['Desc_QuantumOscillator_C'])
  # num_rows += 1
  # end custom factory constraints

  #TODO: create product -> ingredient inequality constraint
  itemsWithNoRecipe = {
    'Desc_QuantumEnergy_C', 'Desc_NitrogenGas_C', 'Desc_LiquidOil_C', 'Desc_OreIron_C', 'Desc_Stone_C',
    'Desc_OreCopper_C', 'Desc_Coal_C', 'Desc_OreGold_C', 'Desc_RawQuartz_C', 'Desc_OreUranium_C', 'Desc_SAM_C', 'Desc_Sulfur_C'}
  # for i in range(len(factories)):
  #   for ingredient, recipe_item_indices in item_to_indices.items():
  #     if ingredient in itemsWithNoRecipe:
  #       continue
  #     for j in recipe_item_indices:
  #       row_indices.extend([num_rows] * (tensor_shape[1]))
  #       col_indices.extend([GetTensorIndex((i + len(resource_nodes), f, j), tensor_shape) for f in range(tensor_shape[1])])
  #       data.extend([-1] * tensor_shape[1])
  #     for j, ratio in ingredient_ratio_pairs_per_product[ingredient]:
  #       row_indices.extend([num_rows] * (tensor_shape[0]))
  #       col_indices.extend([GetTensorIndex((f, i, j), tensor_shape) for f in range(tensor_shape[0])])
  #       data.extend([ratio] * tensor_shape[0])
  #     b_eq.append(0)
  #     num_rows += 1

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
  # expected_resources = {'Desc_OreBauxite_C': -12300, 'Desc_Sulfur_C': -10800, 'Desc_Water_C': -220796.51834329864}
  expected_resources = {'Desc_OreBauxite_C': -12300, 'Desc_Water_C': -230478.90423565393}
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
  # item to edges map
  # resources get item edges only
  # factory to edges list
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
      Vector2(295000, 58000),  # 522
      Vector2(20400, -94300),

      # Vector2(0, 0),

    # Vector2(-45211.86283595924, -112951.80722891566),
    # Vector2(177186.65430954588, 104610.75069508806),

    # Vector2(324525.67191844305, -174119.55514365152),
    # Vector2(7607.784986098239, 199837.81278962002),
    # Vector2(31932.622798887838, -158827.61816496754),
    # Vector2(-193245.87581093604, 19810.009267840593),

    # Vector2(-226605.6533827618, -137279.88878591286),
    # Vector2(-138341.24189063947, 97659.8702502317),
    # Vector2(-22277.01575532899, 239457.8313253012),
    # Vector2(154251.80722891568, 124073.21594068583),
    # Vector2(-30616.960148285434, -158132.5301204819),
    # Vector2(129231.97405004635, -40662.650602409616),
    # Vector2(261281.09360519005, -39967.56255792396),
    # Vector2(314100.7414272474, -231811.8628359592),

    # Vector2(-191160.88971269695, 110866.54309545876),
    # Vector2(-46601.85356811865, 62210.37998146436),
    # Vector2(32627.61816496757, 210264.13345690456),
    # Vector2(-79266.63577386469, 257530.12048192776),
    # Vector2(-284290.2687673772, -91404.07784986094),
    # Vector2(-157106.11677479147, -121987.9518072289),
    # Vector2(-40346.895273401286, -140755.32900834104),
    # Vector2(62512.4189063948, -83758.10936051898),
    # Vector2(157726.78405931417, -9383.688600556052),
    # Vector2(113247.0806302132, 183155.69972196477),
    # Vector2(204291.4735866543, 101830.39851714554),
    # Vector2(231396.2928637627, 19810.009267840593),
    # Vector2(286995.922150139, -90013.90176088968),
    # Vector2(239041.24189063953, -222080.63021316033),
    # Vector2(371090.3614457832, -185240.96385542167),

    # Vector2(-247455.51436515292, 72636.7006487489),
    # Vector2(-271085.35681186285, -38577.386468952696),
    # Vector2(-250930.49119555144, -186631.13994439295),
    # Vector2(-45211.86283595924, -198447.63670064873),
    # Vector2(33322.613531047245, -207483.78127896198),
    # Vector2(144521.87210379983, -207483.78127896198),
    # Vector2(221666.3577386469, -317307.6923076923),
    # Vector2(-39651.89990732161, -10078.776645041653),
    # Vector2(-145986.1909175162, 124073.21594068583),
    # Vector2(89617.23818350327, 104610.75069508806),
    # Vector2(126451.99258572754, 256835.0324374421),
    # Vector2(176491.6589434662, 202618.16496756265),
    # Vector2(214716.40407784982, 135194.62465245603),
    # Vector2(268231.047265987, 113646.89527340134),
    # Vector2(168846.70991658943, -92099.1658943466),
    # Vector2(391245.2270620945, -260310.47265987023),
    # Vector2(290470.89898053755, -168558.85078776642),
    # Vector2(358580.44485634845, -76112.14087117702),

    Vector2(391245.2270620945, -260310.47265987023),
    Vector2(237651.25115848007, -268651.52919369785),
    Vector2(297420.85264133464, -169253.93883225208),
    Vector2(346070.52826691384, -99050.04633920296),
    Vector2(143826.8767377201, -198447.63670064873),
    Vector2(171626.69138090825, -99050.04633920296),
    Vector2(67377.38646895275, -87233.54958294716),
    Vector2(134791.93697868398, -9383.688600556052),
    Vector2(225141.33456904546, -16334.569045412412),
    Vector2(257806.1167747915, 58734.93975903618),
    Vector2(206376.4596848935, 123378.12789620023),
    Vector2(82667.28452270624, 92099.16589434666),
    Vector2(43747.54402224283, 1042.632066728489),
    Vector2(34712.60426320665, -204703.42910101946),
    Vector2(-53551.807228915684, -210959.22150139016),
    Vector2(-43126.8767377201, -140060.2409638554),
    Vector2(65292.400370713614, 206788.69323447638),
    Vector2(-17412.048192771093, 284638.55421686755),
    Vector2(-101506.48748841521, 256835.0324374421),
    Vector2(-47991.844300278055, 197057.4606116775),
    Vector2(-75096.66357738647, 127548.656163114),
    Vector2(-161971.0843373494, 183850.78776645043),
    Vector2(-248150.50973123263, 71941.61260426324),
    Vector2(-266915.3846153846, -41357.738646895275),
    Vector2(-221740.6858202039, -145620.94531974048),
    Vector2(-79266.63577386469, -19114.921223354933),

      # Vector2(59185.2, -8814.81),
      # Oil sites
      # Vector2(-253111, 75555.6),
      # Vector2(-40296.3, -6296.3),
      # Vector2(47222.2, -2518.52),
      # Vector2(178185, 205259),
      # Vector2(260037, 112074),
      # Vector2(169370, -93814.8),
      # Vector2(144815, -209037),
      # Vector2(-40925.9, -195815),
      # Vector2(34629.6, -201481),
      #
      # # Nitrogen sites
      # Vector2(-250593, -183852),
      # Vector2(-146704, 120889),
      # Vector2(84370.4, 105148),
      # Vector2(125926, 256259),
      # Vector2(214074, 135370),
      # Vector2(223519, -311037),
      #
      # # Quartz sites
      # # Vector2(-181333, -142296),
      # # Vector2(-90666.7, 65481.5),
      # # Vector2(34000, 118370),
      # # Vector2(56666.7, 202111),
      # # Vector2(56666.7, -132222),
      # # Vector2(190778, -13851.9),
      # # Vector2(306630, -299704),
      #
      # # Other sites
      # Vector2(-268852, -45963),
      # Vector2(-222259, -133481),
      # Vector2(-144185, -31481.5),
      # Vector2(-50370.4, 248074),
      # Vector2(107667, 177556),
      # Vector2(248074, 44703.7),
      # Vector2(68000, -83740.7),
      # Vector2(-43444.4, -142296),
      # Vector2(-99481.5, -141037),
      # Vector2(388481, -261926),
      # Vector2(302852, -162444),
      # Vector2(125296, -16370.4),
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
  planning_structures = get_planning_structures(max_factory)

  factory_item_edge_list: list[ItemEdge] = []
  i = 0
  for resource, nodes in resource_node_data.items():
    for _ in range(len(nodes)):
      for j in range(len(factories)):
        if j == 1 or (j == 0 and resource not in {'Desc_OreUranium_C', 'Desc_Sulfur_C'}):
          continue
        for resource_edge in planning_structures.resource_edges:
          if resource_edge[0] == resource:
            factory_item_edge_list.append(ItemEdge(resource_edge[0], i, resource_edge[1], j, resource_edge[2]))
      i += 1
  for i, producing_factory_node in enumerate(factories):
    for j, consuming_factory_node in enumerate(factories):
      for factory_edge in planning_structures.factory_edges:
        factory_item_edge_list.append(ItemEdge(factory_edge[0], i + len(resource_nodes), factory_edge[1], j, factory_edge[2]))
  print(len(factory_item_edge_list))
  print(factory_item_edge_list[:100])

  indices_per_consuming_factory_recipe_item = {}
  indices_per_produced_recipe_item = {}
  indices_per_recipe_product = {}
  indices_per_resource_node = {}
  # all factories using ingredient (i) to produce product (p) with recipe (r) being sent to factory (id) with recipe (r)
  for i, factory_item_edge in enumerate(factory_item_edge_list):
    consuming_factory_recipe_item = f'{factory_item_edge.consuming_factory}:{factory_item_edge.consuming_recipe}:{factory_item_edge.item}'
    if consuming_factory_recipe_item not in indices_per_consuming_factory_recipe_item:
      indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item] = []
    indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item].append(i)

    produced_recipe_item = f'{factory_item_edge.producing_factory}:{factory_item_edge.producing_recipe}:{factory_item_edge.item}'
    if produced_recipe_item not in indices_per_produced_recipe_item:
      indices_per_produced_recipe_item[produced_recipe_item] = []
    indices_per_produced_recipe_item[produced_recipe_item].append(i)

    if factory_item_edge.producing_recipe != '':
      recipe_product = f'{factory_item_edge.producing_recipe}:{factory_item_edge.item}'
      if recipe_product not in indices_per_recipe_product:
        indices_per_recipe_product[recipe_product] = []
      indices_per_recipe_product[recipe_product].append(i)

    if factory_item_edge.producing_recipe == '':
      if factory_item_edge.producing_factory not in indices_per_resource_node:
        indices_per_resource_node[factory_item_edge.producing_factory] = []
      indices_per_resource_node[factory_item_edge.producing_factory].append(i)

  print(len(indices_per_consuming_factory_recipe_item))
  print(len(indices_per_produced_recipe_item))
  # print(indices_per_produced_recipe_item.keys())
  planning_structures.indices_per_consuming_factory_recipe_item = indices_per_consuming_factory_recipe_item
  planning_structures.indices_per_produced_recipe_item = indices_per_produced_recipe_item
  planning_structures.indices_per_recipe_product = indices_per_recipe_product
  planning_structures.indices_per_resource_node = indices_per_resource_node

  all_row_indices = []
  all_col_indices = []
  all_data = []
  all_b_eq = []
  next_row_index = 0

  for factory_id in range(len(factory_positions)):
    for recipe_name, recipe in max_factory.items():
      if recipe_name == '_COMMENTS':
        continue
      for product, product_rate in recipe['products'].items():
        for ingredient, ingredient_rate in recipe['ingredients'].items():
          if factory_id < 2 and ingredient in RESOURCE_ITEMS and not (factory_id == 0 and ingredient in {'Desc_OreUranium_C', 'Desc_Sulfur_C'}):
            continue
          consuming_factory_recipe_item = f'{factory_id}:{recipe_name}:{ingredient}'
          # producing_factory = all factories, producing_recipe_edge = child_item, consuming_factory = this.producing_factory, consuming_recipe = this.producing_recipe
          # child constraint
          # all factories using ingredient (i) to produce product (p) with recipe (r) being sent to factory (id) with recipe (r)
          indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
          all_col_indices.extend(indices)
          all_row_indices.extend([next_row_index] * len(indices))
          all_data.extend([product_rate / ingredient_rate] * len(indices))
          produced_recipe_item = f'{factory_id + len(resource_nodes)}:{recipe_name}:{product}'
          indices = planning_structures.indices_per_produced_recipe_item[produced_recipe_item]
          all_col_indices.extend(indices)
          all_row_indices.extend([next_row_index] * len(indices))
          all_data.extend([-1] * len(indices))
          all_b_eq.append(0)
          next_row_index += 1

  for recipe_name, recipe in max_factory.items():
    if recipe_name == '_COMMENTS':
      continue
    for product, rate in recipe['products'].items():
      recipe_product = f'{recipe_name}:{product}'
      if recipe_product not in planning_structures.indices_per_recipe_product:
        continue
      indices = planning_structures.indices_per_recipe_product[recipe_product]
      all_col_indices.extend(indices)
      all_row_indices.extend([next_row_index] * len(indices))
      all_data.extend([1] * len(indices))
      all_b_eq.append(rate)
      next_row_index += 1

  i = 0
  for nodes in resource_node_data.values():
    for node in nodes:
      indices = planning_structures.indices_per_resource_node[i]
      all_col_indices.extend(indices)
      all_row_indices.extend([next_row_index] * len(indices))
      all_data.extend([1] * len(indices))
      all_b_eq.append(node['rate'])
      next_row_index += 1
      i += 1

  # custom factory constraints
  # uranium constraint
  consuming_factory_recipe_item = '0:Alternate: Infused Uranium Cell:Desc_OreUranium_C'
  indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
  all_col_indices.extend(indices)
  all_row_indices.extend([next_row_index] * len(indices))
  all_data.extend([1] * len(indices))
  all_b_eq.append(max_factory['Alternate: Infused Uranium Cell']['ingredients']['Desc_OreUranium_C'])
  next_row_index += 1

  # plutonium constraint
  consuming_factory_recipe_item = '0:Plutonium Fuel Rod (Sink):Desc_PlutoniumFuelRod_C'
  indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
  all_col_indices.extend(indices)
  all_row_indices.extend([next_row_index] * len(indices))
  all_data.extend([1] * len(indices))
  all_b_eq.append(max_factory['Plutonium Fuel Rod (Sink)']['ingredients']['Desc_PlutoniumFuelRod_C'])
  next_row_index += 1
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_OreUranium_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 0, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(2100)
  # num_rows += 1
  #
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_PlutoniumFuelRod_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 0, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['Plutonium Fuel Rod (Sink)']['ingredients']['Desc_PlutoniumFuelRod_C'])
  # num_rows += 1
  #

  # portal constraint
  consuming_factory_recipe_item = '1:Singularity Cell (Portal):Desc_SingularityCell_C'
  indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
  all_col_indices.extend(indices)
  all_row_indices.extend([next_row_index] * len(indices))
  all_data.extend([1] * len(indices))
  all_b_eq.append(max_factory['Singularity Cell (Portal)']['ingredients']['Desc_SingularityCell_C'])
  next_row_index += 1

  # power matrix constraint
  consuming_factory_recipe_item = '1:Alien Power Matrix (Power Augmenter):Desc_AlienPowerFuel_C'
  indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
  all_col_indices.extend(indices)
  all_row_indices.extend([next_row_index] * len(indices))
  all_data.extend([1] * len(indices))
  all_b_eq.append(max_factory['Alien Power Matrix (Power Augmenter)']['ingredients']['Desc_AlienPowerFuel_C'])
  next_row_index += 1

  # warp drive sink constraint
  consuming_factory_recipe_item = '1:Ballistic Warp Drive (Sink):Desc_SpaceElevatorPart_11_C'
  indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
  all_col_indices.extend(indices)
  all_row_indices.extend([next_row_index] * len(indices))
  all_data.extend([1] * len(indices))
  all_b_eq.append(max_factory['Ballistic Warp Drive (Sink)']['ingredients']['Desc_SpaceElevatorPart_11_C'])
  next_row_index += 1

  # warp drive sink constraint
  consuming_factory_recipe_item = '1:Ballistic Warp Drive (Sloopedx4):Desc_SpaceElevatorPart_8_C'
  indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
  all_col_indices.extend(indices)
  all_row_indices.extend([next_row_index] * len(indices))
  all_data.extend([1] * len(indices))
  all_b_eq.append(max_factory['Ballistic Warp Drive (Sloopedx4)']['ingredients']['Desc_SpaceElevatorPart_8_C'])
  next_row_index += 1

  # warp drive sink constraint
  consuming_factory_recipe_item = '1:AI Expansion Server (Sloopedx2):Desc_SpaceElevatorPart_6_C'
  indices = planning_structures.indices_per_consuming_factory_recipe_item[consuming_factory_recipe_item]
  all_col_indices.extend(indices)
  all_row_indices.extend([next_row_index] * len(indices))
  all_data.extend([1] * len(indices))
  all_b_eq.append(max_factory['AI Expansion Server (Sloopedx2)']['ingredients']['Desc_SpaceElevatorPart_6_C'])
  next_row_index += 1
  #
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.recipeName == 'Ballistic Warp Drive (Slooped)' and recipe_item.ingredient == 'Desc_QuantumOscillator_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['Ballistic Warp Drive (Slooped)']['ingredients']['Desc_QuantumOscillator_C'])
  # num_rows += 1
  #
  # # for j, recipe_item in enumerate(recipe_items):
  # #   if recipe_item.ingredient == 'Desc_SpaceElevatorPart_11_C':
  # #     row_indices.extend([num_rows] * tensor_shape[0])
  # #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  # #     data.extend([1] * tensor_shape[0])
  # # b_eq.append(75)
  # # num_rows += 1
  #
  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_SpaceElevatorPart_12_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['AI Expansion Server (Sink)']['ingredients']['Desc_SpaceElevatorPart_12_C'])
  # num_rows += 1

  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.recipeName == 'AI Expansion Server (Slooped)' and recipe_item.ingredient == 'Desc_SpaceElevatorPart_6_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # b_eq.append(max_factory['Ballistic Warp Drive (Slooped)']['ingredients']['Desc_QuantumOscillator_C'])
  # num_rows += 1

  # for j, recipe_item in enumerate(recipe_items):
  #   if recipe_item.ingredient == 'Desc_SpaceElevatorPart_7_C':
  #     row_indices.extend([num_rows] * tensor_shape[0])
  #     col_indices.extend([GetTensorIndex((f, 1, j), tensor_shape) for f in range(tensor_shape[0])])
  #     data.extend([1] * tensor_shape[0])
  # #b_eq.append(133.020559093718)
  # b_eq.append(max_factory['Ballistic Warp Drive (Slooped)']['ingredients']['Desc_QuantumOscillator_C'])
  # num_rows += 1
  # end custom factory constraints

  print(len(all_row_indices))
  print(len(all_col_indices))
  print(len(all_data))
  print(len(all_b_eq))
  print(f'{next_row_index} x {len(factory_item_edge_list)}')

  A_eq = csr_matrix((all_data, (all_row_indices, all_col_indices)),
                           shape=(next_row_index, len(factory_item_edge_list)))

  distances = [f.get_distance(resource_nodes, factory_positions) for f in factory_item_edge_list]

  opt = linprog(c=distances, A_eq=A_eq, b_eq=all_b_eq, # bounds=bounds,
                method='highs')

  #assert False
  # recipe_items = CreateRecipeItemList(max_factory) # resource_recipes + list(max_factory.keys())
  # distances, tensor_shape = CreateDistanceTensor(resource_nodes, factories, recipe_items)
  # # print(distances)
  # ingredient_ratio_pairs_per_product, item_to_indices = GetProductInfo(recipe_items)
  # # A_ub, b_ub, constraints, constraint_eq = CreateConstraints(resource_nodes, factories, recipe_items, ingredient_ratio_pairs_per_product, item_to_indices, tensor_shape)
  # constraints, constraint_eq = CreateConstraints(resource_nodes, factories, recipe_items,
  #                                                            ingredient_ratio_pairs_per_product, item_to_indices,
  #                                                            tensor_shape, max_factory)
  # bounds = [(0, None)] * len(distances.flatten())
  #
  # # opt = linprog(c=distances.flatten(), A_ub=A_ub, b_ub=b_ub, A_eq=constraints, b_eq=constraint_eq, bounds=bounds, method='highs')
  # opt = linprog(c=distances.flatten(), A_eq=constraints, b_eq=constraint_eq, bounds=bounds,
  #               method='highs')
  print(opt)
  # print(opt.x)
  print(len(opt.x))

  used_factory_item_edges = []
  for i, rate in enumerate(opt.x):
    if rate < 1e-8:
      continue
    factory_item_edge_list[i].rate = round(rate, 8)
    used_factory_item_edges.append(factory_item_edge_list[i])

  inputs_per_factory = {}
  outputs_per_factory = {}
  for factory_item_edge in used_factory_item_edges:
    consuming_factory = factory_item_edge.consuming_factory + len(resource_nodes)
    if consuming_factory not in inputs_per_factory:
      inputs_per_factory[consuming_factory] = []
    inputs_per_factory[consuming_factory].append({'id': factory_item_edge.producing_factory, 'recipe': factory_item_edge.consuming_recipe, 'item': factory_item_edge.item, 'rate': float(factory_item_edge.rate)})
    if factory_item_edge.producing_factory not in outputs_per_factory:
      outputs_per_factory[factory_item_edge.producing_factory] = []
    outputs_per_factory[factory_item_edge.producing_factory].append(
      {'id': consuming_factory, 'recipe': factory_item_edge.consuming_recipe,
       'item': factory_item_edge.item, 'rate': float(factory_item_edge.rate)})

  edges = {}
  for i, resource_node in enumerate(resource_nodes):
    edges[f'{i}'] = {'recipeName': f'Resource: {list(resource_node.outputRate.keys())[0]}', 'x': resource_node.position.x,'y': resource_node.position.y, 'input': [], 'output': outputs_per_factory[i]}
  for i, position in enumerate(factory_positions):
    edges[f'{i + len(resource_nodes)}'] = {'recipeName': '', 'x': position.x,'y': position.y, 'input': inputs_per_factory.get(i + len(resource_nodes), []), 'output': outputs_per_factory.get(i + len(resource_nodes), [])}

  print(len(used_factory_item_edges))
  return edges

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
  print(json.dumps(factory_nodes))
  # print(f'{{{',\n'.join([str(f) for f in factory_nodes])}}}')
  satisfactory_data = ReadSatisfactoryData()
  stack_size_map = ParseItems(satisfactory_data)
  transport_map = {
    'DRONES': [0, 0, 0],
    'TRUCKS': [0, 0],
    'TRAINS': [0, 0],
  }
  longest_dist = 7000
  bucket_size = 250
  num_buckets = int(longest_dist // bucket_size)
  total_distance = [0] * num_buckets
  total_stacks = [0] * num_buckets
  for i, factory_node in factory_nodes.items():
    for output in factory_node['output']:
      dist = math.sqrt((factory_nodes[str(output['id'])]['x'] - factory_node['x']) ** 2 + (factory_nodes[str(output['id'])]['y'] - factory_node['y']) ** 2) / 100
      num_stacks = output['rate'] / stack_size_map[output['item']]

      if dist > 2000:
        if num_stacks < 2:
          # station power: 100
          transport_map['DRONES'][0] += 1
          transport_map['DRONES'][1] += 60 * (24000 + 12 * dist) / (10000 * (2 * dist / 100 + 102))
          transport_map['DRONES'][2] += 60 * (24000 + 12 * dist) / (7200 * (2 * dist / 75 + 102))
        else:
          transport_map['TRAINS'][0] += 1
          transport_map['TRAINS'][1] += 63.25 + 50
      else:
        # station power: 20
        transport_map['TRUCKS'][0] += 1
        transport_map['TRUCKS'][1] += 75



      dist_bucket = int(dist // bucket_size)
      total_distance[dist_bucket] += dist
      total_stacks[dist_bucket] += output['rate'] / stack_size_map[output['item']]

  print(transport_map)

  dist_buckets = np.linspace(0, longest_dist, num_buckets+1)
  for i in range(0, len(total_distance)):
    print(f'{dist_buckets[i]}-{dist_buckets[i+1]}: {total_distance[i]:.2f}, {total_stacks[i]:.2f}')

if __name__ == '__main__':
  main()