import json
import math
from enum import Enum
from typing import Union, Any
from dataclasses import dataclass, field
from scipy.optimize import linprog
import numpy as np

import satisfactory_parser

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

# true consts
OVERCLOCK_FACTOR = 1.321928
PUMP_ENERGY = 8

# parameters
CLOCK_SPEED = 2.5
NUM_AUGMENTERS = 0
NUM_FUELED_AUGMENTERS = 1
NUM_WARP_SLOOPS = 15
STABLE_ENERGY = 1261652
MISC_ENERGY = 50000

NUM_PORTALS = 4

# derived consts
NUM_WARP_SLOOPS = min(NUM_WARP_SLOOPS, int((103 - 11 * (NUM_AUGMENTERS + NUM_FUELED_AUGMENTERS)) / 4))
NUM_AI_SLOOPS = math.floor((103 - 11 * (NUM_AUGMENTERS + NUM_FUELED_AUGMENTERS) - NUM_WARP_SLOOPS * 4) / 2)

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

MACHINE_TO_POWER = {
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

NUCLEAR_FUEL = {'Recipe_NuclearFuelRod_C', 'Recipe_Alternate_NuclearFuelRod_1_C', 'Recipe_PlutoniumFuelRod_C', 'Recipe_Alternate_PlutoniumFuelUnit_C', 'Recipe_FicsoniumFuelRod_C'}
NUCLEAR_WASTE = {'Desc_NuclearWaste_C', 'Desc_PlutoniumWaste_C'}

@dataclass
class ResourceNode:
  itemClass: str
  impureRate: int
  impureNodes: int
  normalRate: int
  normalNodes: int
  pureRate: int
  pureNodes: int
  energyCost: int

  def ResourceRate(self):
    return CLOCK_SPEED * (self.impureRate * self.impureNodes + self.normalRate * self.normalNodes + self.pureRate * self.pureNodes)

  def EnergyCost(self):
    return self.energyCost * CLOCK_SPEED ** OVERCLOCK_FACTOR * (self.impureNodes + self.normalNodes + self.pureNodes)

RESOURCES = {
  'Desc_OreIron_C': ResourceNode('Desc_OreIron_C', 120, 39, 240, 42, 480, 46, 45),
  'Desc_Coal_C': ResourceNode('Desc_Coal_C', 120, 15, 240, 31, 480, 16, 45),
  'Desc_NitrogenGas_C': ResourceNode('Desc_NitrogenGas_C', 30, 2, 60, 7, 120, 36, 150),
  'Desc_Sulfur_C': ResourceNode('Desc_Sulfur_C', 120, 6, 240, 5, 480, 5, 45),
  'Desc_SAM_C': ResourceNode('Desc_SAM_C', 120, 10, 240, 6, 480, 3, 45),
  'Desc_OreBauxite_C': ResourceNode('Desc_OreBauxite_C', 120, 5, 240, 6, 480, 6, 45),
  'Desc_OreGold_C': ResourceNode('Desc_OreGold_C', 120, 0, 240, 9, 480, 8, 45),
  'Desc_OreCopper_C': ResourceNode('Desc_OreCopper_C', 120, 13, 240, 29, 480, 13, 45),
  'Desc_Stone_C': ResourceNode('Desc_Stone_C', 120, 15, 240, 49, 480, 30, 45),
  'Desc_RawQuartz_C': ResourceNode('Desc_RawQuartz_C', 120, 3, 240, 7, 480, 7, 45),
  'Desc_OreUranium_C': ResourceNode('Desc_OreUranium_C', 120, 3, 240, 2, 480, 0, 45),
  'Desc_LiquidOil_C': ResourceNode('Desc_LiquidOil_C', 60, 10, 120, 12, 240, 8, 40),
  'Desc_LiquidOil_C2': ResourceNode('Desc_LiquidOil_C', 30, 8, 60, 6, 120, 4, 150),
}

TOTAL_RESOURCE_ENERGY_COST = sum([resource.EnergyCost() for resource in RESOURCES.values()])

@dataclass
class RecipeItem:
  itemClass: str
  rate: float

class RecipeEncodingType(Enum):
  SINK = 0
  INTERMEDIATE = 1
  POWER = 2

@dataclass
class Recipe:
  recipeClass: str
  recipeName: str
  ingredients: list[RecipeItem]
  products: list[RecipeItem]
  energyCost: float

  def Encode(self, itemsDict: dict[str, Any], recipeEncodingType: RecipeEncodingType):
    encodedRecipe = np.zeros(len(itemsDict) + 1)
    for ingredient in self.ingredients:
      if self.recipeClass in NUCLEAR_FUEL and recipeEncodingType != RecipeEncodingType.POWER and ingredient.itemClass == 'Desc_Water_C':
        continue
      encodedRecipe[itemsDict[ingredient.itemClass].index] = -ingredient.rate

    for i, product in enumerate(self.products):
      if (recipeEncodingType != RecipeEncodingType.INTERMEDIATE and i == 0) or (self.recipeClass in NUCLEAR_FUEL and recipeEncodingType != RecipeEncodingType.POWER and product.itemClass in NUCLEAR_WASTE):
        continue
      encodedRecipe[itemsDict[product.itemClass].index] += product.rate
    encodedRecipe[len(itemsDict)] = self.energyCost - (self.products[0].rate * itemsDict[self.products[0].itemClass].energyPerMin if recipeEncodingType == RecipeEncodingType.POWER else 0)
    return encodedRecipe

@dataclass
class Item:
  itemClass: str
  itemName: str
  energyPerMin: float
  pointValue: int
  matterState: MatterState
  index: int = 0
  recipes: list[Recipe] = field(default_factory=lambda: [])


@dataclass
class RecipeGuide:
  recipeClass: str
  recipeName: str
  numBuildings: float
  totalEnergyCost: float
  ingredients: dict[str, float]
  products: dict[str, float]
  byproducts: dict[str, float] = field(default_factory=lambda: {})
  recipeTree: list[Any] = field(default_factory=lambda: [])

  def DisplayTree(self, padding: str = ''):
    print(f'{padding}{self.recipeName}, {self.numBuildings}, {self.totalEnergyCost}, {self.products}')
    padding = padding.replace('┣', '┃').replace('┗', ' ')
    for i, recipe in enumerate(self.recipeTree):
      if i < len(self.recipeTree) - 1:
        recipe.DisplayTree(f'{padding} ┣') # ┣┃┗
      else:
        recipe.DisplayTree(f'{padding} ┗')  # ┣┃┗

  def __str__(self):
    ingredients = ', '.join([f'"{k}": {v}' for k, v in self.ingredients.items()])
    products = ', '.join([f'"{k}": {v}' for k, v in self.products.items()])
    byproducts = ', '.join([f'"{k}": {v}' for k, v in self.byproducts.items()])
    return f'"{self.recipeName}": {{"ingredients": {{{ingredients}}}, "products": {{{products}}}}}'



def ParseItemDesc(itemClasses: list[dict[str, Any]]):
  items = []
  for item in itemClasses:
    items.append(
      Item(
        item['ClassName'],
        item['mDisplayName'],
        float(item['mEnergyValue']) / 60 * (1 if item['mForm'] == 'RF_SOLID' else 1000),
        int(item['mResourceSinkPoints']) if 'mResourceSinkPoints' in item and MATTER_STATE_TABLE[item['mForm']] == MatterState.SOLID else 0,
        MATTER_STATE_TABLE[item['mForm']]))
  return items

def ParseItems(satisfactoryJson: list[dict[str, Any]]):
  items = []
  for data in satisfactoryJson:
    print(data['NativeClass'])
    if data['NativeClass'] in ITEM_DESCRIPTOR_CLASSES:
      print(f'{len(data["Classes"])}: {data["Classes"][0]}')
      items.extend(ParseItemDesc(data["Classes"]))
  print()

  #   for item in data['Classes']:
  #     if 'mDisplayName' in item and item['mDisplayName'] == 'Alien Power Matrix':
  #       print('============================')
  #       print(f'{data["NativeClass"]}{item}')

  return items

def ParseRecipeItems(itemsDict: dict[str, Item], recipe: dict[str, Any], recipeItemDescs: str):
  jsonifiedRecipeItems = (recipeItemDescs[1:-1]
    .replace('(', '{')
    .replace(')', '}')
    .replace('ItemClass=',  '"ItemClass":')
    .replace('Amount=', '"Amount":'))
  jsonRecipeItems = json.loads(f'[{jsonifiedRecipeItems}]')

  if len(jsonRecipeItems) > 0 and jsonRecipeItems[0]['ItemClass'].split('.')[-1][:-1] not in itemsDict:
    return None

  recipeItems = []
  for recipeItem in jsonRecipeItems:
    itemClass = recipeItem['ItemClass'].split('.')[-1][:-1]
    fluidFactor = 1 if itemsDict[itemClass].matterState == MatterState.SOLID else 1000
    recipeItems.append(
      RecipeItem(
        itemClass,
        CLOCK_SPEED * recipeItem['Amount'] / float(recipe['mManufactoringDuration']) * 60.0 / fluidFactor))

  return recipeItems

def ParseRecipes(satisfactoryJson: list[dict[str, Any]], itemsDict: dict[str, Item]):
  recipes = []
  for data in satisfactoryJson:
    if data['NativeClass'] == '/Script/CoreUObject.Class\'/Script/FactoryGame.FGRecipe\'':
      print(data['NativeClass'])
      print(f'{len(data["Classes"])}: {data["Classes"][0]}')
      for recipe in data['Classes']:
        products = ParseRecipeItems(itemsDict, recipe, recipe['mProduct'])
        if products is None:
          continue

        ingredients = ParseRecipeItems(itemsDict, recipe, recipe['mIngredients'])
        if ingredients is None:
          continue

        machines = recipe['mProducedIn'].replace('(', '').replace(')', '').split(',')
        for machine in machines:
          machine = machine.split('.')[-1][:-1]
          if machine != 'BP_WorkBenchComponent_C':
            break

        if MACHINE_TO_POWER[machine] != 0:
          energyCost = MACHINE_TO_POWER[machine]
        else:
          energyCost = (2 * float(recipe['mVariablePowerConsumptionConstant']) + float(recipe['mVariablePowerConsumptionFactor'])) / 2
        recipes.append(Recipe(recipe['ClassName'], recipe['mDisplayName'], ingredients, products,
                              energyCost * CLOCK_SPEED ** OVERCLOCK_FACTOR))

        if recipes[-1].recipeClass in {'Recipe_NuclearFuelRod_C', 'Recipe_Alternate_NuclearFuelRod_1_C'}:
          recipes[-1].ingredients.append(RecipeItem('Desc_Water_C', CoolentRate(itemsDict, recipes[-1])))
          recipes[-1].products.append(RecipeItem('Desc_NuclearWaste_C', recipes[-1].products[0].rate / .2 * 10))
        if recipes[-1].recipeClass in {'Recipe_PlutoniumFuelRod_C', 'Recipe_Alternate_PlutoniumFuelUnit_C'}:
          recipes[-1].ingredients.append(RecipeItem('Desc_Water_C', CoolentRate(itemsDict, recipes[-1])))
          recipes[-1].products.append(RecipeItem('Desc_PlutoniumWaste_C', recipes[-1].products[0].rate / .1 * 1))
        if recipes[-1].recipeClass in {'Recipe_FicsoniumFuelRod_C'}:
          recipes[-1].ingredients.append(RecipeItem('Desc_Water_C', CoolentRate(itemsDict, recipes[-1])))
        if recipes[-1].recipeClass in {'Recipe_SpaceElevatorPart_12_C', 'Recipe_SpaceElevatorPart_11_C'}:
          sloopedProducts = [RecipeItem(product.itemClass, product.rate * 2) for product in recipes[-1].products]
          sloopedRecipe = Recipe(f'{recipes[-1].recipeClass}2', f'{recipes[-1].recipeName} (Slooped)', recipes[-1].ingredients,
                                 sloopedProducts, recipes[-1].energyCost)
          sloopedRecipe.energyCost *= 4
          recipes.append(sloopedRecipe)

  recipes.append(Recipe('Recipe_Water_C', 'Water', [],
                        [RecipeItem(r'Desc_Water_C', CLOCK_SPEED * 120)], 20 * CLOCK_SPEED ** OVERCLOCK_FACTOR + 1 * PUMP_ENERGY))
  print()
  return recipes

def CoolentRate(itemsDict: dict[str, Item], recipe: Recipe):
  energyPerRecipe = itemsDict[recipe.products[0].itemClass].energyPerMin * recipe.products[0].rate
  energyPerReactor = 2500 * CLOCK_SPEED
  coolentPerReactor = 240 * CLOCK_SPEED
  return energyPerRecipe / energyPerReactor * coolentPerReactor

def SomersloopPower(initialPower: float, numSloops: int, numChargedSloops: int):
  return (initialPower + ((numSloops + numChargedSloops) * 500)) * (.1 * numSloops + .3 * numChargedSloops)

def EncodeSatisfactory(itemsDict: dict[str, Item], orderedRecipes: list[Recipe], debug=False):
  encodedRecipes = []
  encodedGoal = []
  bounds = []

  # SINK
  for recipe in orderedRecipes:
    if itemsDict[recipe.products[0].itemClass].pointValue == 0:
      continue
    encodedRecipes.append(recipe.Encode(itemsDict, RecipeEncodingType.SINK))
    # Must only use the first product as dark matter is a side product of AI Expansion Servers
    encodedGoal.append(itemsDict[recipe.products[0].itemClass].pointValue * recipe.products[0].rate)
    if recipe.recipeClass == 'Recipe_SpaceElevatorPart_11_C2':
      bounds.append((0, NUM_WARP_SLOOPS))
    elif recipe.recipeClass == 'Recipe_SpaceElevatorPart_12_C2':
      bounds.append((0, NUM_AI_SLOOPS))
    else:
      bounds.append((0, None))

  # INTERMEDIATE
  for recipe in orderedRecipes:
    encodedRecipes.append(recipe.Encode(itemsDict, RecipeEncodingType.INTERMEDIATE))
    encodedGoal.append(0)
    bounds.append((0, None))

  # POWER
  for recipe in orderedRecipes:
    if itemsDict[recipe.products[0].itemClass].energyPerMin == 0:
      continue
    encodedRecipes.append(recipe.Encode(itemsDict, RecipeEncodingType.POWER))
    encodedGoal.append(0)
    bounds.append((0, None))

  encodedGoal = -1 * np.array(encodedGoal)

  encodedRecipes = np.array(encodedRecipes).transpose()

  encodedLimits = np.zeros(len(itemsDict) + 1)

  # Resource Constraints
  for resource in RESOURCES:
    if resource in {'Desc_LiquidOil_C', 'Desc_LiquidOil_C2'}:
      continue
    encodedLimits[itemsDict[resource].index] = -RESOURCES[resource].ResourceRate()
  encodedLimits[itemsDict['Desc_LiquidOil_C'].index] = -1 * (
        RESOURCES['Desc_LiquidOil_C'].ResourceRate() + RESOURCES['Desc_LiquidOil_C2'].ResourceRate())

  # Power Constraint
  encodedLimits[-1] = -1 * TOTAL_RESOURCE_ENERGY_COST + SomersloopPower(
    STABLE_ENERGY, NUM_AUGMENTERS, NUM_FUELED_AUGMENTERS) - MISC_ENERGY

  # Alien Fuel Constraint
  encodedLimits[itemsDict['Desc_AlienPowerFuel_C'].index] = 5 * NUM_FUELED_AUGMENTERS

  # Portal constraint
  encodedLimits[itemsDict['Desc_SingularityCell_C'].index] = 2 * NUM_PORTALS

  # Transport Constraint
  encodedLimits[itemsDict['Desc_PackagedRocketFuel_C'].index] = 417

  if debug:
    print(encodedGoal)
    print(encodedRecipes.shape)
    print(encodedRecipes)
    print(encodedLimits)
    print(bounds)
    print()

  return encodedGoal, encodedRecipes, encodedLimits, bounds

def GetMaxPointsGuide(itemsDict: dict[str, Item], orderedRecipes: list[Recipe], printRecipeGuide: bool = True, debug: bool = False):
  encodedGoal, encodedRecipes, encodedLimits, bounds = EncodeSatisfactory(itemsDict, orderedRecipes)
  opt = linprog(c=encodedGoal, A_eq=encodedRecipes, b_eq=encodedLimits, bounds=bounds, method='highs')
  if debug:
    print(opt)
    print(opt.x)
    print(len(opt.x))

  totalPowerProduced = 0
  i = 0

  recipeGuides = {}

  # for item in itemsDict.values():
  #   if item.itemClass == 'Desc_Water_C':
  #     print(encodedRecipes[item.index][:])
  #     print(opt.x * encodedRecipes[item.index][:])

  print('SINK:')
  for recipe in orderedRecipes:
    if itemsDict[recipe.products[0].itemClass].pointValue == 0:
      continue
    if opt.x[i] != 0:
      recipeGuides[recipe.recipeClass] = RecipeGuide(recipe.recipeClass, recipe.recipeName, opt.x[i],
                                                     opt.x[i] * recipe.energyCost,
                                                     {ingredient.itemClass: ingredient.rate * opt.x[i] for ingredient in
                                                      recipe.ingredients if not (recipe.recipeClass in NUCLEAR_FUEL and ingredient.itemClass == 'Desc_Water_C')},
                                                     {product.itemClass: product.rate * opt.x[i] for product in
                                                      recipe.products if product.itemClass not in NUCLEAR_WASTE})
      if printRecipeGuide:
        print(recipeGuides[recipe.recipeClass])
    i += 1

  print('INTERMEDIATE:')
  for recipe in orderedRecipes:
    if opt.x[i] != 0:
      recipeGuides[recipe.recipeClass] = RecipeGuide(recipe.recipeClass, recipe.recipeName, opt.x[i],
                                                     opt.x[i] * recipe.energyCost,
                                                     {ingredient.itemClass: ingredient.rate * opt.x[i] for ingredient in
                                                      recipe.ingredients},
                                                     {product.itemClass: product.rate * opt.x[i] for product in
                                                      recipe.products})
      if printRecipeGuide:
        print(recipeGuides[recipe.recipeClass])
    i += 1

  print('POWER:')
  for recipe in orderedRecipes:
    if itemsDict[recipe.products[0].itemClass].energyPerMin == 0:
      continue
    if opt.x[i] != 0:
      recipeGuides[recipe.recipeClass] = RecipeGuide(recipe.recipeClass, recipe.recipeName, opt.x[i],
                                                     opt.x[i] * recipe.energyCost,
                                                     {ingredient.itemClass: ingredient.rate * opt.x[i] for ingredient in
                                                      recipe.ingredients},
                                                     {product.itemClass: product.rate * opt.x[i] for product in
                                                      recipe.products})
      if printRecipeGuide:
        print(recipeGuides[recipe.recipeClass])
      totalPowerProduced += itemsDict[recipe.products[0].itemClass].energyPerMin * recipe.products[0].rate * opt.x[i]
    i += 1
  print(f'Points/min: {-opt.fun}')
  print(f'TotalPowerProduced: {totalPowerProduced}')
  print(
    f'TotalBoostedPower: {totalPowerProduced + SomersloopPower(totalPowerProduced, NUM_AUGMENTERS, NUM_FUELED_AUGMENTERS)}')
  print(
    f'TotalPowerConsumed: {sum([recipeGuide.totalEnergyCost for recipeGuide in recipeGuides.values()]) + TOTAL_RESOURCE_ENERGY_COST + MISC_ENERGY}')
  print()

  return recipeGuides, totalPowerProduced

SKIPPED_INGREDIENTS = {'Desc_Rubber_C', 'Desc_Plastic_C', 'Desc_DarkEnergy_C'}
CHOOSE_INGREDIENT_RECIPE = {'Recipe_SulfuricAcid_C:Desc_Water_C': 'Recipe_NonFissileUranium_C', 'Recipe_Alternate_CircuitBoard_1_C:Desc_Silica_C': 'Recipe_AluminaSolution_C', 'Recipe_Alternate_IngotSteel_2_C:Desc_CompactedCoal_C': 'Recipe_RocketFuel_C'} #, 'Recipe_Alternate_Silica_Distilled_C:Desc_Water_C': 'Recipe_Alternate_Silica_Distilled_C'}

def AddRecipeTree(recipeGuide: RecipeGuide, itemsDict: dict[str, Item], recipeGuides: dict[str, RecipeGuide]):
  ingredientGuides = []
  # print(f'Breaking down {recipeGuide.recipeName}:')
  # print(recipeGuide)
  for ingredient, totalIngredientRate in recipeGuide.ingredients.items():
    if ingredient in SKIPPED_INGREDIENTS:
      print(f'Skipping {ingredient}')
      ingredientGuides.append(RecipeGuide(ingredient, itemsDict[ingredient].itemName, -1,
                  -1,
                  {},
                  {ingredient: totalIngredientRate}, byproducts={'isByproduct': 1}))
      continue
    while totalIngredientRate > 1e-6:
      validRecipes = sorted([recipe for recipe in itemsDict[ingredient].recipes if recipe.recipeClass in recipeGuides and recipeGuides[recipe.recipeClass].products[ingredient] > 1e-6], key=lambda recipe: -recipeGuides[recipe.recipeClass].products[ingredient])
      recipeIndex = 0
      if len(validRecipes) > 1:
        recipeIngredientPair = f'{recipeGuide.recipeClass}:{ingredient}'
        recipeClasses = [recipe.recipeClass for recipe in validRecipes]
        if recipeIngredientPair in CHOOSE_INGREDIENT_RECIPE and CHOOSE_INGREDIENT_RECIPE[recipeIngredientPair] in recipeClasses:
          recipeIndex = recipeClasses.index(CHOOSE_INGREDIENT_RECIPE[recipeIngredientPair])
        # elif ingredient == 'Desc_DarkMatter_C':
        #   print(f'Breaking down {recipeGuide.recipeName}:')
        #   print(recipeGuide)
        #   print(f'  Select recipe for {itemsDict[ingredient].itemName}:')
        #   for i, recipe in enumerate(validRecipes):
        #     print(f'    {i}: {recipeGuides[recipe.recipeClass]}')
        #   recipeIndex = int(input('    index: '))
      if len(validRecipes) == 0:
        if ingredient not in RESOURCES:
          print(f'Breaking down {recipeGuide.recipeName}:')
          print(recipeGuide)
          print(f'ERROR: No recipe for {ingredient} missing {totalIngredientRate}')
        else:
          ingredientGuides.append(RecipeGuide(ingredient, itemsDict[ingredient].itemName, -1,
                                              -1,
                                              {},
                                              {ingredient: totalIngredientRate}, byproducts={'isByproduct': 1}))
        break

      validRecipe = validRecipes[recipeIndex]
      ingredientRecipeGuide = recipeGuides[validRecipe.recipeClass]
      ingredientRateForRecipe = [product.rate for product in validRecipe.products if product.itemClass == ingredient][0]
      if ingredient in ingredientRecipeGuide.byproducts and ingredientRecipeGuide.byproducts[ingredient] > 1e-6:
        totalIngredientRateForRecipe = min(totalIngredientRate, ingredientRecipeGuide.byproducts[ingredient])
        totalIngredientRate -= totalIngredientRateForRecipe
        ingredientRecipeGuide.products[ingredient] -= totalIngredientRateForRecipe
        ingredientRecipeGuide.byproducts[ingredient] -= totalIngredientRateForRecipe
        numBuildings = totalIngredientRateForRecipe / ingredientRateForRecipe
        ingredientGuide = RecipeGuide(validRecipe.recipeClass, validRecipe.recipeName, numBuildings,
                                      numBuildings * validRecipe.energyCost,
                                      {ingredient.itemClass: ingredient.rate * numBuildings for ingredient in
                                       validRecipe.ingredients},
                                      {product.itemClass: product.rate * numBuildings for product in
                                       validRecipe.products}, byproducts={'isByproduct': 1})
      else:
        totalIngredientRateForRecipe = min(totalIngredientRate, ingredientRecipeGuide.products[ingredient])
        totalIngredientRate -= totalIngredientRateForRecipe
        ingredientRecipeGuide.products[ingredient] -= totalIngredientRateForRecipe
        numBuildings = totalIngredientRateForRecipe / ingredientRateForRecipe
        for product in validRecipe.products:
          if product.itemClass == ingredient:
            continue
          if product.itemClass not in ingredientRecipeGuide.byproducts:
            ingredientRecipeGuide.byproducts[product.itemClass] = 0
          ingredientRecipeGuide.byproducts[product.itemClass] += numBuildings * product.rate
        ingredientGuide = RecipeGuide(validRecipe.recipeClass, validRecipe.recipeName, numBuildings,
                                      numBuildings * validRecipe.energyCost,
                                      {ingredient.itemClass: ingredient.rate * numBuildings for ingredient in
                                       validRecipe.ingredients},
                                      {product.itemClass: product.rate * numBuildings for product in
                                       validRecipe.products}, byproducts={'isByproduct': 0})
      ingredientGuides.append(ingredientGuide)

      # print(f'  {ingredientGuide}')
  #print()

  recipeGuide.recipeTree = ingredientGuides
  for ingredientGuide in ingredientGuides:
    if ingredientGuide.byproducts['isByproduct'] == 1:
      continue
    AddRecipeTree(ingredientGuide, itemsDict, recipeGuides)
  return ingredientGuides

def main():
  satisfactoryJson = satisfactory_parser.read_satisfactory_data()

  items = ParseItems(satisfactoryJson)
  itemsDict = {item.itemClass: item for item in items if item.itemClass not in ITEMS_TO_FILTER} # and item.itemClass in {'Desc_OreIron_C', 'Desc_Coal_C', 'Desc_Sulfur_C', 'Desc_SAM_C', 'Desc_OreBauxite_C', 'Desc_OreGold_C', 'Desc_OreCopper_C', 'Desc_Stone_C', 'Desc_OreUranium_C'}}

  recipes = ParseRecipes(satisfactoryJson, itemsDict)

  for recipe in recipes:
    for product in recipe.products:
      itemsDict[product.itemClass].recipes.append(recipe)

  print(len(itemsDict))
  orderedItems = sorted([item for item in itemsDict.values()], key=lambda item: item.pointValue)
  for i, item in enumerate(orderedItems):
    itemsDict[item.itemClass].index = i
    print(item)
  print()

  print(len(recipes))
  orderedRecipes = []
  for item in orderedItems:
    for recipe in item.recipes:
      if recipe.products[0].itemClass != item.itemClass:
        continue
      orderedRecipes.append(recipe)
      print(recipe)
  print()

  recipeGuides, totalPowerProduced = GetMaxPointsGuide(itemsDict, orderedRecipes)

  global STABLE_ENERGY
  while (totalPowerProduced - STABLE_ENERGY) < 0 or (totalPowerProduced - STABLE_ENERGY) > 500:
    STABLE_ENERGY = (STABLE_ENERGY + totalPowerProduced) / 2 - (1000 if totalPowerProduced - STABLE_ENERGY < 0 else 0)
    print(f'Stable Energy: {STABLE_ENERGY}')
    recipeGuides, totalPowerProduced = GetMaxPointsGuide(itemsDict, orderedRecipes, False)



  AddRecipeTree(recipeGuides['Recipe_SpaceElevatorPart_11_C2'], itemsDict, recipeGuides)
  # recipeGuides['Recipe_SpaceElevatorPart_12_C2'].products['Desc_SpaceElevatorPart_12_C'] -= recipeGuides['Recipe_SpaceElevatorPart_12_C2'].byproducts['Desc_SpaceElevatorPart_12_C']
  # recipeGuides['Recipe_SpaceElevatorPart_12_C2'].byproducts['Desc_DarkEnergy_C'] = recipeGuides['Recipe_SpaceElevatorPart_12_C2'].products['Desc_DarkEnergy_C']
  # recipeGuides['Recipe_SpaceElevatorPart_12_C2'].numBuildings = recipeGuides['Recipe_SpaceElevatorPart_12_C2'].products['Desc_SpaceElevatorPart_12_C'] / itemsDict['Desc_SpaceElevatorPart_12_C'].recipes[1].products[0].rate
  # recipeGuides['Recipe_SpaceElevatorPart_12_C2'].totalEnergyCost = recipeGuides['Recipe_SpaceElevatorPart_12_C2'].numBuildings * itemsDict['Desc_SpaceElevatorPart_12_C'].recipes[1].energyCost
  # recipeGuides['Recipe_SpaceElevatorPart_12_C2'].ingredients = {ingredient.itemClass: recipeGuides['Recipe_SpaceElevatorPart_12_C2'].numBuildings * ingredient.rate for ingredient in itemsDict['Desc_SpaceElevatorPart_12_C'].recipes[1].ingredients}
  AddRecipeTree(recipeGuides['Recipe_SpaceElevatorPart_12_C2'], itemsDict, recipeGuides)
  AddRecipeTree(recipeGuides['Recipe_SpaceElevatorPart_7_C'], itemsDict, recipeGuides)
  AddRecipeTree(recipeGuides['Recipe_PlutoniumFuelRod_C'], itemsDict, recipeGuides)
  AddRecipeTree(recipeGuides['Recipe_AlienPowerFuel_C'], itemsDict, recipeGuides)

  recipeGuides['Recipe_SingularityCell_C'].numBuildings = recipeGuides['Recipe_SingularityCell_C'].products['Desc_SingularityCell_C'] / itemsDict['Desc_SingularityCell_C'].recipes[0].products[0].rate
  recipeGuides['Recipe_SingularityCell_C'].totalEnergyCost = recipeGuides['Recipe_SingularityCell_C'].numBuildings * itemsDict['Desc_SingularityCell_C'].recipes[0].energyCost
  recipeGuides['Recipe_SingularityCell_C'].ingredients = {ingredient.itemClass: recipeGuides['Recipe_SingularityCell_C'].numBuildings * ingredient.rate for ingredient in itemsDict['Desc_SingularityCell_C'].recipes[0].ingredients}

  AddRecipeTree(recipeGuides['Recipe_SingularityCell_C'], itemsDict, recipeGuides)
  AddRecipeTree(recipeGuides['Recipe_PackagedRocketFuel_C'], itemsDict, recipeGuides)

  recipeGuides['Recipe_SpaceElevatorPart_11_C2'].DisplayTree()
  recipeGuides['Recipe_SpaceElevatorPart_12_C2'].DisplayTree()
  recipeGuides['Recipe_SpaceElevatorPart_7_C'].DisplayTree()
  recipeGuides['Recipe_PlutoniumFuelRod_C'].DisplayTree()
  recipeGuides['Recipe_AlienPowerFuel_C'].DisplayTree()
  recipeGuides['Recipe_SingularityCell_C'].DisplayTree()
  recipeGuides['Recipe_PackagedRocketFuel_C'].DisplayTree()

  # already created by making compact coal
  # AddRecipeTree(recipeGuides['Recipe_RocketFuel_C'], itemsDict, recipeGuides)
  # AddRecipeTree(recipeGuides['Recipe_Alternate_RocketFuel_Nitro_C'], itemsDict, recipeGuides)
  # recipeGuides['Recipe_RocketFuel_C'].DisplayTree()
  # recipeGuides['Recipe_Alternate_RocketFuel_Nitro_C'].DisplayTree()
  for recipeGuide in recipeGuides.values():
    if len(recipeGuide.recipeTree) != 0:
      continue
    #print(recipeGuide)
    for byproduct, rate in recipeGuide.byproducts.items():
      if rate > 1e-6:
        print(f'ERROR: excess {byproduct}')

if __name__ == '__main__':
  main()