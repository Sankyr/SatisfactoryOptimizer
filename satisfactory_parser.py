import json
from enum import Enum
from typing import Any
from dataclasses import dataclass, field
import numpy as np

OVERCLOCK_FACTOR = 1.321928


class MatterState(Enum):
  SOLID = 0
  LIQUID = 1
  GAS = 2


MATTER_STATE_TABLE = {
  'RF_SOLID': MatterState.SOLID,
  'RF_LIQUID': MatterState.LIQUID,
  'RF_GAS': MatterState.GAS
}

ITEM_DESCRIPTOR_CLASSES = {
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGItemDescriptor\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGItemDescriptorNuclearFuel\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGItemDescriptorBiomass\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGItemDescriptorPowerBoosterFuel\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGResourceDescriptor\'',
  '/Script/CoreUObject.Class\'/Script/FactoryGame.FGPowerShardDescriptor\'',
}

# TODO: pull from satisfactory.json directly
MACHINE_TO_POWER = {
  'Build_MinerMk1_C': 5,
  'Build_MinerMk2_C': 15,
  'Build_MinerMk3_C': 45,
  'Build_FrackingSmasher_C': 150,
  'Build_SmelterMk1_C': 4,
  'Build_ConstructorMk1_C': 4,
  'Build_Packager_C': 10,
  'Build_AssemblerMk1_C': 15,
  'Build_FoundryMk1_C': 16,
  'Build_WaterPump_C': 20,
  'Build_OilRefinery_C': 30,
  'Build_ManufacturerMk1_C': 55,
  'Build_Blender_C': 75,
  'Build_HadronCollider_C': 0,
  'Build_QuantumEncoder_C': 0,
  'Build_Converter_C': 0
}

MACHINE_TO_SLOOPS = {
  'Build_SmelterMk1_C': 1,
  'Build_ConstructorMk1_C': 1,
  'Build_AssemblerMk1_C': 2,
  'Build_FoundryMk1_C': 2,
  'Build_OilRefinery_C': 2,
  'Build_ManufacturerMk1_C': 4,
  'Build_Blender_C': 4,
  'Build_HadronCollider_C': 4,
  'Build_QuantumEncoder_C': 4,
  'Build_Converter_C': 2
}


@dataclass
class Generator:
  class_name: str
  display_name: str
  water_rate: float
  energy_rate: float
  accepted_fuels: list[str]


GENERATORS = [
  Generator('Build_GeneratorCoal_C', 'Coal Gen', 45, 75,
            ['Desc_Coal_C', 'Desc_CompactedCoal_C', 'Desc_PetroleumCoke_C']),
  Generator('Build_GeneratorFuel_C', 'Fuel Gen', 0, 250,
            ['Desc_LiquidFuel_C', 'Desc_LiquidTurboFuel_C', 'Desc_RocketFuel_C', 'Desc_IonizedFuel_C']),
  Generator('Build_GeneratorNuclear_C', 'Nuclear Plant', 240, 2500,
            ['Desc_NuclearFuelRod_C', 'Desc_PlutoniumFuelRod_C', 'Desc_FicsoniumFuelRod_C'])
]


@dataclass
class ResourceNode:
  item_class: str
  display_name: str
  impure_nodes: int
  normal_nodes: int
  pure_nodes: int


RESOURCES = [
  ResourceNode('Desc_OreIron_C', 'Iron', 39, 42, 46),
  ResourceNode('Desc_Coal_C', 'Coal', 15, 31, 16),
  ResourceNode('Desc_NitrogenGas_C', 'Nitrogen Well', 2, 7, 36),
  ResourceNode('Desc_Sulfur_C', 'Sulfur', 6, 5, 5),
  ResourceNode('Desc_SAM_C', 'SAM', 10, 6, 3),
  ResourceNode('Desc_OreBauxite_C', 'Bauxite', 5, 6, 6),
  ResourceNode('Desc_OreGold_C', 'Caterium', 0, 9, 8),
  ResourceNode('Desc_OreCopper_C', 'Copper', 13, 29, 13),
  ResourceNode('Desc_Stone_C', 'Limestone', 15, 49, 30),
  ResourceNode('Desc_RawQuartz_C', 'Quartz', 3, 7, 7),
  ResourceNode('Desc_OreUranium_C', 'Uranium', 3, 2, 0),
  ResourceNode('Desc_LiquidOil_C', 'Oil', 10, 12, 8),
  ResourceNode('Desc_LiquidOil_C', 'Oil Well', 8, 6, 4),
]


@dataclass
class RecipeItem:
  itemClass: str
  rate: float


@dataclass
class Recipe:
  recipe_class: str
  recipe_name: str
  ingredients: list[RecipeItem]
  products: list[RecipeItem]
  energy_cost: float
  num_sloops: int
  points_per_min: float

  amount: float = 0
  index: int = 0

  def encode(self, items_dict: dict[str, Any], clock_speed: float):
    # convert recipe into list of rates per item +
    # an additional value for power usage/production & num_sloops respectively
    if '(Sink)' in self.recipe_name:
      clock_speed = 1
    encoded_recipe = np.zeros(len(items_dict) + 2)
    for ingredient in self.ingredients:
      encoded_recipe[items_dict[ingredient.itemClass].index] -= clock_speed * ingredient.rate

    for i, product in enumerate(self.products):
      encoded_recipe[items_dict[product.itemClass].index] += clock_speed * product.rate

    if self.energy_cost > 0:
      encoded_recipe[-2] = self.energy_cost * clock_speed
    else:
      encoded_recipe[-2] = self.energy_cost * clock_speed ** OVERCLOCK_FACTOR

    encoded_recipe[-1] = self.num_sloops

    return encoded_recipe

  def to_json(self, clock_speed: float):
    if '(Sink)' in self.recipe_name:
      clock_speed = 1

    return json.dumps({
      self.recipe_name: {
        'ingredients': {i.itemClass: clock_speed * i.rate * self.amount for i in self.ingredients},
        'products': {p.itemClass: clock_speed * p.rate * self.amount for p in self.products},
        }
      })[1:-1]


@dataclass
class Item:
  item_class: str
  item_name: str
  energy_per_min: float
  point_value: int
  matter_state: MatterState

  index: int = 0
  created_by: list[str] = field(default_factory=lambda: [])
  used_in: list[str] = field(default_factory=lambda: [])


def read_satisfactory_data() -> list[dict[str, Any]]:
  with open('satisfactory_data.json', 'r', encoding="utf-16") as f:
    json_str = f.read()
  return json.loads(json_str)


def parse_item_desc(item_classes: list[dict[str, Any]]) -> list[Item]:
  items = []
  for item in item_classes:
    is_solid = item['mForm'] == 'RF_SOLID'
    items.append(
      Item(
        item['ClassName'],
        item['mDisplayName'],
        float(item['mEnergyValue']) / 60 * (1 if is_solid else 1000),
        int(item['mResourceSinkPoints']) if 'mResourceSinkPoints' in item and is_solid else 0,
        MATTER_STATE_TABLE[item['mForm']]))
  return items


def parse_items(satisfactory_json: list[dict[str, Any]]) -> list[Item]:
  items = []
  for data in satisfactory_json:
    if data['NativeClass'] in ITEM_DESCRIPTOR_CLASSES:
      items.extend(parse_item_desc(data["Classes"]))
  return items


def create_sink_recipes(items: list[Item]) -> list[Recipe]:
  recipes = []
  for item in items:
    if item.point_value == 0:
      continue
    recipes.append(Recipe(
      f'{item.item_class}_Sink_C', f'{item.item_name} (Sink)', [RecipeItem(item.item_class, 1)],
      [], 0, 0, item.point_value))
  return recipes


def create_resource_recipes() -> list[Recipe]:
  recipes = []
  for resource in RESOURCES:
    name = resource.display_name
    mining_speed = 240
    energy_cost = MACHINE_TO_POWER['Build_MinerMk3_C']
    if 'Well' in name:
      mining_speed = 60
      energy_cost = MACHINE_TO_POWER['Build_FrackingSmasher_C']
    recipes.append(Recipe(
      f'{name}_Impure_C', f'{name} (Impure)', [],
      [RecipeItem(resource.item_class, 0.5 * mining_speed)], -energy_cost, 0, 0))
    recipes.append(Recipe(
      f'{name}_Normal_C', f'{name} (Normal)', [],
      [RecipeItem(resource.item_class, 1 * mining_speed)], -energy_cost, 0, 0))
    recipes.append(Recipe(
      f'{name}_Pure_C', f'{name} (Pure)', [],
      [RecipeItem(resource.item_class, 2 * mining_speed)], -energy_cost, 0, 0))
  recipes.append(Recipe('Recipe_Water_C', 'Water', [], [RecipeItem('Desc_Water_C', 120)], -20, 0, 0))
  return recipes


def create_power_recipes(items_dict: dict[str, Item]) -> list[Recipe]:
  recipes = []
  for generator in GENERATORS:
    for fuel in generator.accepted_fuels:
      ingredients = [RecipeItem(fuel, generator.energy_rate / items_dict[fuel].energy_per_min)]
      if generator.water_rate != 0:
        ingredients.append(RecipeItem('Desc_Water_C', generator.water_rate))

      products = []
      if fuel == 'Desc_NuclearFuelRod_C':
        products.append(RecipeItem('Desc_NuclearWaste_C', 10))
      if fuel == 'Desc_PlutoniumFuelRod_C':
        products.append(RecipeItem('Desc_PlutoniumWaste_C', 1))
      recipes.append(Recipe(
        f'{generator.class_name}_{fuel}', f'{generator.display_name} ({fuel})', ingredients,
        products, generator.energy_rate, 0, 0))
  return recipes


def parse_production_machine(produced_in_str: str) -> str | None:
  machines = produced_in_str.replace('(', '').replace(')', '').split(',')
  for machine in machines:
    machine = machine.split('.')[-1][:-1]
    if machine != 'BP_WorkBenchComponent_C':
      return machine
  return None


def parse_recipe_items(items_dict: dict[str, Item], recipe: dict[str, Any], recipe_item_desc_json: str) -> list[RecipeItem] | None:
  jsonified_recipe_items = (recipe_item_desc_json[1:-1]
    .replace('(', '{')
    .replace(')', '}')
    .replace('ItemClass=',  '"ItemClass":')
    .replace('Amount=', '"Amount":'))
  json_recipe_items = json.loads(f'[{jsonified_recipe_items}]')

  if len(json_recipe_items) > 0 and json_recipe_items[0]['ItemClass'].split('.')[-1][:-1] not in items_dict:
    return None

  recipe_items = []
  for recipe_item in json_recipe_items:
    item_class = recipe_item['ItemClass'].split('.')[-1][:-1]
    fluid_factor = 1 if items_dict[item_class].matter_state == MatterState.SOLID else 1000
    recipe_items.append(
      RecipeItem(item_class, recipe_item['Amount'] / float(recipe['mManufactoringDuration']) * 60.0 / fluid_factor))

  return recipe_items


def parse_recipes(satisfactoryJson: list[dict[str, Any]], itemsDict: dict[str, Item]) -> list[Recipe]:
  recipes = []
  for data in satisfactoryJson:
    if data['NativeClass'] != '/Script/CoreUObject.Class\'/Script/FactoryGame.FGRecipe\'':
      continue

    for recipe in data['Classes']:
      products = parse_recipe_items(itemsDict, recipe, recipe['mProduct'])
      if products is None:
        continue

      ingredients = parse_recipe_items(itemsDict, recipe, recipe['mIngredients'])
      if ingredients is None:
        continue

      machine = parse_production_machine(recipe['mProducedIn'])

      if MACHINE_TO_POWER[machine] != 0:
        energyCost = MACHINE_TO_POWER[machine]
      else:
        energyCost = (2 * float(recipe['mVariablePowerConsumptionConstant']) + float(recipe['mVariablePowerConsumptionFactor'])) / 2
      recipes.append(Recipe(recipe['ClassName'], recipe['mDisplayName'], ingredients, products,
                            -energyCost, 0, 0))

      if machine in MACHINE_TO_SLOOPS:
        slooped_products = [RecipeItem(p.itemClass, 2 * p.rate) for p in products]
        recipes.append(Recipe(f'{recipe['ClassName']}_Slooped', f'{recipe['mDisplayName']} (Slooped)',
                              ingredients, slooped_products, -4 * energyCost, MACHINE_TO_SLOOPS[machine], 0))
  return recipes
