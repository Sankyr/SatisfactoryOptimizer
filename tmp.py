from typing import Any

import pygame
import pygame_gui

import json
import random

from dataclasses import dataclass, field

import factory_router

pos = (500, 500)

Vector2 = pygame.math.Vector2
Color = pygame.Color

DRAG_COEFF = 1

images = {}

current_node = None

MAP_WIDTH = 457200 - -340000
MAP_HEIGHT = 335000 - -345000
MAP_SCALE = 1.12

EDGE_ID = 0

nodes = []
edges = []

simulation_enabled = False

@dataclass
class Edge:
  id: int
  recipe: str
  item: str
  rate: float
  color: pygame.Color = field(default_factory=lambda: Color(255, 255, 255, 64))

@dataclass
class Node:
  label: str
  inputs: list[Edge]
  outputs: list[Edge]
  color: pygame.Color
  position: pygame.math.Vector2

  products: dict[str, float] = field(default_factory=lambda: {})
  image: pygame.Surface | None = None
  mass: float = 1
  locked: bool = True
  visible: bool = True
  force: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))
  velocity: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))

  # for RK4
  p0: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))
  v0: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))
  a0: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))
  a1: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))
  a2: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))
  a3: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))

  def set_products(self):
    for output in self.outputs:
      if output.item not in self.products:
        self.products[output.item] = 0
      self.products[output.item] += output.rate

def ReadNodeData():
  # Open and read the JSON file
  with open('node_locations.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)


def ReadMaxFactoryRoute():
  # Open and read the JSON file
  with open('max_factory_route.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)


def update_pos_leap_frog(surface, dt):
  for edge in edges:
    edge.node1.force = edge.node1.force - edge.k * (edge.node1.position - edge.node2.position)
    edge.node2.force = edge.node2.force - edge.k * (edge.node2.position - edge.node1.position)

  # half step
  for node in nodes:
    acceleration = node.force / node.mass
    node.velocity = node.velocity + .5 * acceleration * dt
    node.position = node.position + node.velocity * dt
    pygame.draw.circle(surface, node.color, node.position, 10)
    node.force = Vector2(0, 0)

  for edge in edges:
    edge.node1.force = edge.node1.force - edge.k * (edge.node1.position - edge.node2.position)
    edge.node2.force = edge.node2.force - edge.k * (edge.node2.position - edge.node1.position)

  for node in nodes:
    acceleration = node.force / node.mass
    node.velocity = node.velocity + .5 * acceleration * dt

  for edge in edges:
    pygame.draw.line(surface, Color(255, 255, 255), edge.node1.position, edge.node2.position)

def update_pos_rk4(surface, dt):
  dt = min(dt, 1/60)
  k = 1 / 900
  for node in nodes:
    for edge in node.outputs:
      node.force = node.force - k * edge.rate * (node.position - nodes[edge.id].position)
      nodes[edge.id].force = nodes[edge.id].force - k * edge.rate * (nodes[edge.id].position - node.position)

  for node in nodes:
    if node.locked:
      continue
    node.p0 = node.position
    node.v0 = node.velocity
    try:
      drag_force = DRAG_COEFF * node.velocity.magnitude() ** 2 * node.velocity.normalize()
    except:
      drag_force = Vector2(0, 0)
    node.a0 = node.force / node.mass - drag_force
    node.velocity = node.v0 + .5 * dt * node.a0
    node.position = node.p0 + .5 * dt * node.velocity
    node.force = Vector2(0, 0)

  for node in nodes:
    for edge in node.outputs:
      node.force = node.force - k * edge.rate * (node.position - nodes[edge.id].position)
      nodes[edge.id].force = nodes[edge.id].force - k * edge.rate * (nodes[edge.id].position - node.position)

  for node in nodes:
    if node.locked:
      continue
    try:
      drag_force = DRAG_COEFF * node.velocity.magnitude() ** 2 * node.velocity.normalize()
    except:
      drag_force = Vector2(0, 0)
    node.a0 = node.force / node.mass - drag_force
    node.velocity = node.v0 + .5 * dt * node.a1
    node.position = node.p0 + .5 * dt * node.velocity
    node.force = Vector2(0, 0)

  for node in nodes:
    for edge in node.outputs:
      node.force = node.force - k * edge.rate * (node.position - nodes[edge.id].position)
      nodes[edge.id].force = nodes[edge.id].force - k * edge.rate * (nodes[edge.id].position - node.position)

  for node in nodes:
    if node.locked:
      continue
    try:
      drag_force = DRAG_COEFF * node.velocity.magnitude() ** 2 * node.velocity.normalize()
    except:
      drag_force = Vector2(0, 0)
    node.a0 = node.force / node.mass - drag_force
    node.velocity = node.v0 + dt * node.a2
    node.position = node.p0 + dt * node.velocity
    node.force = Vector2(0, 0)

  for node in nodes:
    for edge in node.outputs:
      node.force = node.force - k * edge.rate * (node.position - nodes[edge.id].position)
      nodes[edge.id].force = nodes[edge.id].force - k * edge.rate * (nodes[edge.id].position - node.position)

  for node in nodes:
    if node.locked:
      node.velocity = Vector2(0, 0)
      continue
    try:
      drag_force = DRAG_COEFF * node.velocity.magnitude() ** 2 * node.velocity.normalize()
    except:
      drag_force = Vector2(0, 0)
    node.a0 = node.force / node.mass - drag_force
    node.velocity = node.v0 + 1 / 6.0 * dt * (node.a0 + node.a1 + node.a2 + node.a3)
    node.position = node.p0 + dt * node.v0 + 1 / 6.0 * dt ** 2 * (node.a0 + node.a1 + node.a2)
    node.force = Vector2(0, 0)


def get_color(resource: str, rate: int):
  if resource == 'Desc_LiquidOil_C':
    rate *= 2

  if resource in {'Desc_LiquidOil_C2', 'Desc_NitrogenGas_C'}:
    rate *= 4

  if rate == 300:
    return Color(210, 52, 48)
  elif rate == 600:
    return Color(242, 100, 24)
  elif rate == 1200:
    return Color(128, 177, 57)

  return Color(128, 128, 128)

def update(surface, alpha_surface, dt: float):
  for node in nodes:
    node.force = Vector2(0, 0)

  if simulation_enabled:
    update_pos_rk4(surface, dt)

  for node in nodes:
    for edge in node.outputs:
      rect = pygame.draw.line(alpha_surface, edge.color, node.position, nodes[edge.id].position, 2)
      surface.blit(alpha_surface, rect, rect)
      alpha_surface.fill(Color(0, 0, 0, 0), rect)

  for i, node in enumerate(nodes):
    if not node.visible:
      continue
    if node.image is None:
      pygame.draw.circle(surface, node.color, node.position, 16)
      text_surface = font.render(str(i), True, (0, 0, 0))
      # 5. Draw (blit) the text surface onto the screen at (X, Y) coordinates
      surface.blit(text_surface, node.position - pygame.Vector2(text_surface.size)  / 2)
    else:
      pygame.draw.circle(surface, Color(255, 255, 255, 255), node.position, 17)
      pygame.draw.circle(surface, node.color, node.position, 16)
      if node.image is not None:
        surface.blit(node.image, node.position.elementwise() - 16)

  global item_rate
  surface.blit(item_rate, (1300, 400))

def load_icon(filename):
  return pygame.transform.scale(pygame.image.load(filename), (32, 32))  # .convert_alpha()

def load_icons():
  images = {}
  images['Desc_RawQuartz_C'] = load_icon("Images/Raw_Quartz.webp")
  images['Desc_OreCopper_C'] = load_icon("Images/Copper_Ore.png")
  images['Desc_Stone_C'] = load_icon("Images/Limestone.png")
  images['Desc_OreIron_C'] = load_icon("Images/Iron_Ore.webp")
  images['Desc_Coal_C'] = load_icon("Images/Coal.png")
  images['Desc_OreGold_C'] = load_icon("Images/Caterium_Ore.png")
  images['Desc_Sulfur_C'] = load_icon("Images/Sulfur.png")
  images['Desc_SAM_C'] = load_icon("Images/SAM.webp")
  images['Desc_OreBauxite_C'] = load_icon("Images/Bauxite.webp")
  images['Desc_OreUranium_C'] = load_icon("Images/Uranium.png")
  images['Desc_LiquidOil_C'] = load_icon("Images/Crude_Oil.webp")
  images['Desc_NitrogenGas_C'] = load_icon("Images/Nitrogen_Gas.webp")
  return images

def WorldToScreen(worldCoords: Vector2) -> Vector2:
  return 1080 * ((worldCoords / MAP_HEIGHT).elementwise() + .5)

def ScreenToWorld(screenCoords: Vector2) -> Vector2:
  return ((screenCoords / 1080).elementwise() - .5) * MAP_HEIGHT
  #return ((x / 1080 - .5) * MAP_HEIGHT, (y / 1080 - .5) * MAP_HEIGHT)

def handle_mouse(x, y):
  global current_node
  print(f'(x, y): {(x, y)}, {ScreenToWorld(Vector2(x, y))})')
  # check click in reverse order to match display order
  for i in range(len(nodes)-1, -1, -1):
    node = nodes[i]
    if not node.visible:
      continue
    if (node.position - Vector2(x, y)).magnitude() < 17:
      print(i)
      print(node)

      global text_input
      if text_input.text in node.products:
        print(f'{text_input.text}: {node.products[text_input.text]}')

      for item_input in node.inputs:
        print(f'I: {item_input}')
      for output in node.outputs:
        print(f'O: {output}')
      current_node = node
      break

def handle_key(key):
  global current_node, simulation_enabled

  if key == pygame.K_SPACE:
    simulation_enabled = not simulation_enabled

  if key == pygame.K_a:
    resource = list(current_node.outputRate.keys())[0]
    current_node.allocate(resource, allocations[resource][0])

  if key == pygame.K_s:
    current_node.split()

def handle_item_lookup(item: str, nodes: list[Node]):
  if item == '':
    for node in nodes:
      node.visible = True
  else:
    total_rate = 0
    for node in nodes:
      if item not in node.products:
        node.visible = False
      else:
        node.visible = True
        total_rate += node.products[item]
    global item_rate
    item_rate = font.render(f'{item}: {total_rate}/min', True, (255, 255, 255))

def createFactoryNodes(factory_nodes) -> list[Node]:
  nodes = []
  for _, factory_node in factory_nodes.items():
    if factory_node['recipeName']:
      resource = factory_node['recipeName'].split(': ')[1]
      node = Node(factory_node['recipeName'], [], [], Color(255, 255, 255), WorldToScreen(Vector2(factory_node['x'], factory_node['y'])), locked=True, image=images[resource])
      total_output = 0
      for output in factory_node['output']:
        total_output += output['rate']
      node.color = get_color(resource, total_output)
    else:
      node = Node('', [], [], Color(255, 255, 255), WorldToScreen(Vector2(factory_node['x'], factory_node['y'])), locked=False)

    for input in factory_node['input']:
      node.inputs.append(Edge(input['id'], input['recipe'], input['item'], input['rate']))
    for output in factory_node['output']:
      node.outputs.append(Edge(output['id'], output['recipe'], output['item'], output['rate']))
    node.set_products()
    nodes.append(node)
  return nodes

#
# def ReadSatisfactoryData():
#   # Open and read the JSON file
#   with open('satisfactory_data.json', 'r', encoding="utf-16") as f:
#     jsonStr = f.read()
#   return json.loads(jsonStr)
#
#
# def ParseRecipeItems(itemsDict: dict[str, Item], recipe: dict[str, Any], recipeItemDescs: str):
#   jsonifiedRecipeItems = (recipeItemDescs[1:-1]
#     .replace('(', '{')
#     .replace(')', '}')
#     .replace('ItemClass=',  '"ItemClass":')
#     .replace('Amount=', '"Amount":'))
#   jsonRecipeItems = json.loads(f'[{jsonifiedRecipeItems}]')
#
#   if len(jsonRecipeItems) > 0 and jsonRecipeItems[0]['ItemClass'].split('.')[-1][:-1] not in itemsDict:
#     return None
#
#   recipeItems = []
#   for recipeItem in jsonRecipeItems:
#     itemClass = recipeItem['ItemClass'].split('.')[-1][:-1]
#     fluidFactor = 1 if itemsDict[itemClass].matterState == MatterState.SOLID else 1000
#     recipeItems.append(
#       RecipeItem(
#         itemClass,
#         CLOCK_SPEED * recipeItem['Amount'] / float(recipe['mManufactoringDuration']) * 60.0 / fluidFactor))
#
#   return recipeItems
#
# def ParseRecipes(satisfactoryJson: list[dict[str, Any]], itemsDict: dict[str, Item]):
#   recipe_to_product = []
#   for data in satisfactoryJson:
#     if data['NativeClass'] == '/Script/CoreUObject.Class\'/Script/FactoryGame.FGRecipe\'':
#       for recipe in data['Classes']:
#         if
#         products = ParseRecipeItems(itemsDict, recipe, recipe['mProduct'])
#   print()
#   return recipes


def main():
  # initialize the pygame module
  pygame.init()

  global font
  font = pygame.font.SysFont("Arial", 18)

  global images
  images = load_icons()
  images['map'] = pygame.image.load("Images/map.jpg")  # .convert_alpha()
  images['map'] = pygame.transform.scale(images['map'], MAP_SCALE*Vector2(1080 * images['map'].width / images['map'].height, 1080))

  global nodes
  # node_locations = ReadNodeData()
  # factory_nodes = factory_router.GetRoute(node_locations)
  factory_nodes = ReadMaxFactoryRoute()
  nodes = createFactoryNodes(factory_nodes)

  # satisfactory_data = ReadSatisfactoryData()


  # for i, node in enumerate(nodes):
  #   if i == 0:
  #     continue
  #   edges.append(Edge(node, nodes[0], node.rate / 1200.0))

  # load and set the logo
  # logo = pygame.image.load("logo32x32.png")
  # pygame.display.set_icon(logo)
  pygame.display.set_caption('Satisfactory Factory Explorer')

  screen_size = (1920, 1080)
  window_surface = pygame.display.set_mode(screen_size, display=0)

  background = pygame.Surface(window_surface.get_size())
  background.fill(Color(0, 0, 0, 255))
  alpha_background = pygame.Surface(window_surface.get_size(), pygame.SRCALPHA)
  alpha_background.fill(Color(0, 0, 0, 0))

  manager = pygame_gui.UIManager(screen_size)

  hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1300, 200), (100, 50)),  text='Say Hello', manager=manager)

  global text_input
  text_input = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((1300, 275), (200, 50)),
    manager=manager
  )

  global item_rate
  item_rate = font.render('', True, (255, 255, 255))

  # define a variable to control the main loop
  clock = pygame.time.Clock()
  running = True

  # main loop
  while running:
    time_delta = clock.tick(60) / 1000.0
    # event handling, gets all event from the event queue
    for event in pygame.event.get():
      # only do something if the event is of type QUIT
      if event.type == pygame.QUIT:
        # change the value to False, to exit the main loop
        running = False
      if event.type == pygame_gui.UI_BUTTON_PRESSED:
        if event.ui_element == hello_button:
          print('Hello World!')
      # handle MOUSEBUTTONUP
      if event.type == pygame.MOUSEBUTTONUP:
        handle_mouse(*pygame.mouse.get_pos())
      if not text_input.is_focused and event.type == pygame.KEYDOWN:
        handle_key(event.key)
      if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
        print(f"User entered: {event.text}")  # Get the text
        if event.ui_element == text_input:
          handle_item_lookup(event.text, nodes)

      manager.process_events(event)

    manager.update(time_delta)
    update(background, alpha_background, time_delta)

    window_surface.blit(background, (0, 0))
    # window_surface.blit(alpha_background, (0, 0))
    background.fill(Color(0, 0, 0, 255))
    background.blit(images['map'], (10, ((1-MAP_SCALE) / 2 * 1080)))
    manager.draw_ui(window_surface)

    pygame.display.update()

if __name__ == '__main__':
  main()