import json
import math
from enum import Enum
from typing import Union, Any
from dataclasses import dataclass, field
from scipy.optimize import linprog
import numpy as np

import satisfactory_parser


Recipe = satisfactory_parser.Recipe
Item = satisfactory_parser.Item

CLOCK_SPEED = 2.5
TOTAL_SLOOPS = 106 - 2 - 1
GEOTHERMAL_POWER = 7100
NUM_IONIZED_FUEL = 0
NUM_AUGMENTERS = 0
NUM_FUELED_AUGMENTERS = 0
NUM_PORTALS = 0
SURPLUS_POWER = 0 #  100000

ITEMS_TO_FILTER = {
  'Desc_HazmatFilter_C',
  'Desc_Filter_C',
  'Desc_Gunpowder_C',
  'Desc_GunpowderMK2_C',
  'Desc_Crystal_C',
  'Desc_Crystal_mk2_C',
  'Desc_Crystal_mk3_C',
  'Desc_AlienProtein_C',
  'Desc_AlienDNACapsule_C',
  'Desc_WAT1_C',
  'Desc_WAT2_C',
  'Desc_XmasStar_C',
  'Desc_XmasBallCluster_C',
  'Desc_XmasWreath_C',
  'Desc_CandyCane_C',
  'Desc_Snow_C',
  'Desc_XmasBow_C',
  'Desc_XmasBall3_C',
  'Desc_XmasBall4_C',
  'Desc_XmasBranch_C',
  'Desc_XmasBall1_C',
  'Desc_XmasBall2_C',
  'Desc_XmasDataCartridge4_C',
  'Desc_XmasDataCartridge3_C',
  'Desc_XmasDataCartridge2_C',
  'Desc_XmasDataCartridge1_C',
  'Desc_Gift_C',
  'Desc_Biofuel_C',
  'Desc_PackagedBiofuel_C',
  'Desc_GenericBiomass_C',
  'Desc_Leaves_C',
  'Desc_Mycelia_C',
  'Desc_Wood_C',
  'Desc_LiquidBiofuel_C',
  'Desc_Fabric_C',
  'Desc_HogParts_C',
  'Desc_SpitterParts_C',
  'Desc_StingerParts_C',
  'Desc_HatcherParts_C',
}


def encode_satisfactory(items_dict: dict[str, Item], recipes: list[Recipe], clock_speed: float, boost_bonus: float = 0, debug=False):
  recipes_dict = {recipe.recipe_name: recipe for recipe in recipes}
  encoded_goals = np.array([-recipe.points_per_min for recipe in recipes])
  encoded_recipes = np.array([recipe.encode(items_dict, clock_speed) for recipe in recipes]).transpose()
  encoded_limits = np.zeros(len(items_dict) + 2)
  encoded_limits[items_dict['Desc_PackagedIonizedFuel_C'].index] = NUM_IONIZED_FUEL
  encoded_limits[items_dict['Desc_SingularityCell_C'].index] = 2 * NUM_PORTALS
  encoded_limits[items_dict['Desc_AlienPowerFuel_C'].index] = 5 * NUM_FUELED_AUGMENTERS
  encoded_limits[-2] = -GEOTHERMAL_POWER - 500 * (NUM_AUGMENTERS + NUM_FUELED_AUGMENTERS) - boost_bonus + SURPLUS_POWER
  encoded_limits[-1] = TOTAL_SLOOPS - 10 * (NUM_AUGMENTERS + NUM_FUELED_AUGMENTERS)

  bounds: list[tuple[float, float | None]] = [(0, None)] * (len(recipes))
  for resource in satisfactory_parser.RESOURCES:
    bounds[recipes_dict[f'{resource.display_name} (Impure)'].index] = (0, resource.impure_nodes)
    bounds[recipes_dict[f'{resource.display_name} (Normal)'].index] = (0, resource.normal_nodes)
    bounds[recipes_dict[f'{resource.display_name} (Pure)'].index] = (0, resource.pure_nodes)
  # bounds[recipes_dict['Fuel Gen (Desc_IonizedFuel_C)'].index] = (1, 1)
  # bounds[recipes_dict['Nuclear Plant (Desc_FicsoniumFuelRod_C)'].index] = (1, 1)

  if debug:
    print(encoded_goals)
    print(encoded_recipes.shape)
    print(encoded_recipes)
    print(encoded_limits)
    print(bounds)
    print()

  return encoded_goals, encoded_recipes, encoded_limits, bounds


def get_max_points_guide(items_dict: dict[str, Item], recipes: list[Recipe], clock_speed: float, boost_bonus: float = 0, printRecipeGuide: bool = True, debug: bool = False):
  encodedGoal, encodedRecipes, encodedLimits, bounds = encode_satisfactory(items_dict, recipes, clock_speed, boost_bonus, debug)
  opt = linprog(c=encodedGoal, A_eq=encodedRecipes, b_eq=encodedLimits, bounds=bounds, method='highs')
  if debug:
    print(opt)
    print(opt.x)
    print(len(opt.x))

  final_recipes = []
  total_power_consumed = 0
  total_power_produced = 0
  total_sloops_used = 0
  for i, recipe in enumerate(recipes):
    if opt.x[i] == 0:
      continue
    recipe.amount = opt.x[i]
    if recipe.energy_cost > 0:
      total_power_produced += recipe.amount * recipe.energy_cost * clock_speed
    else:
      total_power_consumed += recipe.amount * recipe.energy_cost * clock_speed ** satisfactory_parser.OVERCLOCK_FACTOR
    total_sloops_used += recipe.amount * recipe.num_sloops
    if debug:
      print(recipe)
    final_recipes.append(recipe)
  print()

  total_power = total_power_produced + GEOTHERMAL_POWER + 500 * (NUM_AUGMENTERS + NUM_FUELED_AUGMENTERS)
  total_boosted_power = total_power * (1 + .1 * NUM_AUGMENTERS + .3 * NUM_FUELED_AUGMENTERS)

  print(f'Points/min: {-opt.fun}')
  print(f'TotalSloopsUsed: {total_sloops_used}')
  print(f'TotalPowerProduced: {total_power} ({total_power_produced} + {GEOTHERMAL_POWER + 500 * (NUM_AUGMENTERS + NUM_FUELED_AUGMENTERS)})')
  print(f'TotalBoostedPower: {total_boosted_power}')
  print(f'TotalPowerConsumed: {total_power_consumed}')
  print()

  return final_recipes, total_power, total_boosted_power, total_power_consumed

def main():
  satisfactory_json = satisfactory_parser.read_satisfactory_data()

  items = satisfactory_parser.parse_items(satisfactory_json)
  items = [item for item in items if item.item_class not in ITEMS_TO_FILTER]
  items.sort(key=lambda item: item.point_value)
  for i, item in enumerate(items):
    item.index = i
    print(item)
  print()

  items_dict = {item.item_class: item for item in items}

  recipes = []
  recipes.extend(satisfactory_parser.create_sink_recipes(items))
  recipes.extend(satisfactory_parser.create_resource_recipes())
  recipes.extend(satisfactory_parser.parse_recipes(satisfactory_json, items_dict))
  recipes.extend(satisfactory_parser.create_power_recipes(items_dict))
  for i, recipe in enumerate(recipes):
    recipe.index = i
    print(recipe)
  print()

  boost_bonus = 0
  # encode_satisfactory(items_dict, recipes, clock_speed=2.5, debug=True)
  for i in range(0, 20):
    final_recipes, total_power, total_boosted_power, total_power_consumed = get_max_points_guide(items_dict, recipes, clock_speed=CLOCK_SPEED, boost_bonus=boost_bonus, debug=False)
    if 500 + SURPLUS_POWER > total_boosted_power + total_power_consumed > 0 + SURPLUS_POWER:
      break
    boost_bonus += total_boosted_power + (total_power_consumed - SURPLUS_POWER)
    print(f'boost_bonus: {boost_bonus}')

  for recipe in final_recipes:
    print(recipe)

  print('{')
  for recipe in final_recipes:
    print(f'{recipe.to_json(CLOCK_SPEED)},')
  print('}')


if __name__ == '__main__':
  main()