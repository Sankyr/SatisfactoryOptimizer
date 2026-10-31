from typing import Any

import pygame
import pygame_gui

import json
import random
import math

from dataclasses import dataclass, field

import factory_router
import satisfactory_parser

from scipy.spatial import Voronoi


Vector2 = pygame.math.Vector2
Color = pygame.Color

MAP_HEIGHT = 335000 - -345000

SCREEN_SIZE = (1920, 1080)
MAP_SIZE = (-324600, -375000, 425300, 375000)

RESOURCE_ITEMS = {'Desc_OreIron_C', 'Desc_Coal_C', 'Desc_Sulfur_C', 'Desc_SAM_C', 'Desc_OreBauxite_C', 'Desc_OreGold_C', 'Desc_OreCopper_C', 'Desc_Stone_C', 'Desc_RawQuartz_C', 'Desc_OreUranium_C', 'Desc_LiquidOil_C', 'Desc_NitrogenGas_C'}



@dataclass
class Edge:
  id: int
  recipe: str
  item: str
  rate: float
  color: pygame.Color = field(default_factory=lambda: Color(255, 255, 255, 64))

@dataclass
class Node:
  id: int
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
  rim_color: Color = field(default_factory=lambda: Color(255, 255, 255, 255))
  force: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))
  velocity: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))
  dominant_force: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))

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

  def update_dominant_force(self, nodes: list[Any]):
    neighbor_nodes = [(math.atan2(nodes[edge.id].position.y, nodes[edge.id].position.x), Vector2(nodes[edge.id].position)) for edge in self.inputs if edge.item in RESOURCE_ITEMS]
    if len(neighbor_nodes) < 2:
      return

    neighbor_nodes.sort(key=lambda key: key[0])
    neighbor_nodes.extend([(n[0] + 2 * math.pi, n[1]) for n in neighbor_nodes])

    max_sum = Vector2(0, 0)
    right = 1
    distance_sum = neighbor_nodes[0][1]
    for left in range(int(len(neighbor_nodes) / 2)):
      while neighbor_nodes[right][0] - neighbor_nodes[left][0] < math.pi:
        distance_sum += neighbor_nodes[right][1]
        right += 1
      if distance_sum.magnitude() > max_sum.magnitude():
        max_sum = distance_sum
      distance_sum -= neighbor_nodes[left][1]

    self.dominant_force = max_sum.normalize()

def load_icon(filename):
  return pygame.transform.smoothscale(pygame.image.load(filename), (24, 24))  # .convert_alpha()


def load_icons():
  return {
    'Desc_RawQuartz_C': load_icon("Images/Raw_Quartz.webp"),
    'Desc_OreCopper_C': load_icon("Images/Copper_Ore.png"), 'Desc_Stone_C': load_icon("Images/Limestone.png"),
    'Desc_OreIron_C': load_icon("Images/Iron_Ore.webp"), 'Desc_Coal_C': load_icon("Images/Coal.png"),
    'Desc_OreGold_C': load_icon("Images/Caterium_Ore.png"), 'Desc_Sulfur_C': load_icon("Images/Sulfur.png"),
    'Desc_SAM_C': load_icon("Images/SAM.webp"), 'Desc_OreBauxite_C': load_icon("Images/Bauxite.webp"),
    'Desc_OreUranium_C': load_icon("Images/Uranium.png"),
    'Desc_LiquidOil_C': load_icon("Images/Crude_Oil.webp"),
    'Desc_NitrogenGas_C': load_icon("Images/Nitrogen_Gas.webp")
  }


def read_max_factory_route():
  with open('test_route.json', 'r', encoding="utf8") as f:
  # with open('near_max_factory_with_extras_route.json', 'r', encoding="utf8") as f:
  # with open('true_max_factory_route.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)


def get_color(resource: str, rate: int):
  if resource == 'Desc_LiquidOil_C':
    rate *= 2

  if resource in {'Desc_LiquidOil_C2', 'Desc_NitrogenGas_C'}:
    rate *= 4

  if abs(rate - 300) < 1e-4:
    return Color(210, 52, 48)
  elif abs(rate - 600) < 1e-4:
    return Color(242, 100, 24)
  elif abs(rate - 1200) < 1e-4:
    return Color(128, 177, 57)

  return Color(128, 128, 128)


def create_factory_nodes(factory_nodes, images: dict[str, Any]) -> list[Node]:
  nodes = []
  for id, factory_node in factory_nodes.items():
    if factory_node['recipeName']:
      resource = factory_node['recipeName'].split(': ')[1]
      node = Node(int(id), factory_node['recipeName'], [], [], Color(255, 255, 255), world_to_screen(Vector2(factory_node['x'], factory_node['y'])), locked=True, image=images[resource])
      total_output = 0
      for output in factory_node['output']:
        total_output += output['rate']
      node.color = get_color(resource, total_output)
    else:
      node = Node(int(id), '', [], [], Color(255, 255, 255), world_to_screen(Vector2(factory_node['x'], factory_node['y'])), locked=int(id) in {522, 523})

    for input in factory_node['input']:
      node.inputs.append(Edge(input['id'], input['recipe'], input['item'], input['rate']))
    for output in factory_node['output']:
      node.outputs.append(Edge(output['id'], output['recipe'], output['item'], output['rate']))
    node.set_products()
    nodes.append(node)
  return nodes


def world_to_screen(world_coords: Vector2) -> Vector2:
  map_dim = (1024, 1024)
  return ((world_coords.elementwise() - Vector2(MAP_SIZE[0], MAP_SIZE[1])).elementwise() *
          Vector2((int(map_dim[0] * SCREEN_SIZE[1] / map_dim[1]) - 1) / (MAP_SIZE[2] - MAP_SIZE[0]),
                  (SCREEN_SIZE[1] - 1) / (MAP_SIZE[3] - MAP_SIZE[1])))


def screen_to_world(screen_coords: Vector2) -> Vector2:
  map_dim = (1024, 1024)
  return (screen_coords.elementwise() *
          Vector2((MAP_SIZE[2] - MAP_SIZE[0]) / (int(map_dim[0] * SCREEN_SIZE[1] / map_dim[1]) - 1),
                  (MAP_SIZE[3] - MAP_SIZE[1]) / (SCREEN_SIZE[1] - 1)) + Vector2(MAP_SIZE[0], MAP_SIZE[1]))


def ReadSatisfactoryData():
  with open('satisfactory_data.json', 'r', encoding="utf-16") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)


def ParseRecipeItems(recipe_item_desc: str, duration: float, non_solids: set[str]) -> list[str]:
  jsonifiedRecipeItems = (recipe_item_desc[1:-1]
    .replace('(', '{')
    .replace(')', '}')
    .replace('ItemClass=',  '"ItemClass":')
    .replace('Amount=', '"Amount":'))
  jsonRecipeItems = json.loads(f'[{jsonifiedRecipeItems}]')

  recipe_items = []
  for recipeItem in jsonRecipeItems:
    item_class = recipeItem['ItemClass'].split('.')[-1][:-1]
    fluid_factor = 1000 if item_class in non_solids else 1
    recipe_items.append((item_class, recipeItem['Amount'] / duration * 60.0 / fluid_factor))

  return recipe_items

def get_product_to_recipe_map(satisfactoryJson: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
  non_solids = set()
  for item in satisfactory_parser.parse_items(satisfactoryJson):
    if item.matter_state == satisfactory_parser.MatterState.GAS or item.matter_state == satisfactory_parser.MatterState.LIQUID:
      non_solids.add(item.item_class)

  product_to_recipe = {}
  ingredient_to_recipe = {}
  recipe_ingredient_to_amount = {}
  for data in satisfactoryJson:
    if data['NativeClass'] == '/Script/CoreUObject.Class\'/Script/FactoryGame.FGRecipe\'':
      for recipe in data['Classes']:
        recipe_name = recipe['mDisplayName']
        for product, rate in ParseRecipeItems(recipe['mProduct'], float(recipe['mManufactoringDuration']), non_solids):
          if product not in product_to_recipe:
            product_to_recipe[product] = set()
          if product in {'Desc_SpaceElevatorPart_11_C'}:
            product_to_recipe[product].add(f'{recipe_name} (Slooped)')
          product_to_recipe[product].add(recipe_name)
        for ingredient, rate in ParseRecipeItems(recipe['mIngredients'], float(recipe['mManufactoringDuration']), non_solids):
          if ingredient not in ingredient_to_recipe:
            ingredient_to_recipe[ingredient] = set()
          ingredient_to_recipe[ingredient].add(recipe_name)
          recipe_ingredient_to_amount[f'{recipe_name}:{ingredient}'] = rate
  return product_to_recipe, ingredient_to_recipe, recipe_ingredient_to_amount


class FactoryExplorer:
  def __init__(self):
    self.font = None
    self.images = None
    self.nodes: list[Node] = []
    self.current_node: Node | None = None
    self.window_surface = None
    self.ui_manager = None
    self.product_search_box = None
    self.running: bool = True
    self.clock = None
    self.frame_buffer = None
    self.alpha_frame_buffer = None
    self.item_rate_label = None
    self.product_to_recipes = None
    self.ingredient_to_recipes = None
    self.recipe_ingredient_to_amount = None
    self.node_info_box = None
    self.search_history: list[str] = []
    self.backward_button = None
    self.forward_button = None
    self.history_index = -1
    self.run_simulation = False
    self.voronoi_points = []

  def run(self):
    self.init()
    self.mainLoop()
    self.cleanup()

  def init(self):
    pygame.init()
    pygame.display.set_caption('Satisfactory Factory Explorer')
    self.window_surface = pygame.display.set_mode(SCREEN_SIZE, display=0)

    self.font = pygame.font.SysFont("Arial", 16)

    self.images = load_icons()
    map_img = pygame.image.load("Images/map.jpg")  # .convert_alpha()

    map_edge = map_img.width * SCREEN_SIZE[1] / map_img.height

    self.images['map'] = pygame.transform.scale(
      map_img, Vector2(map_edge, SCREEN_SIZE[1]))

    self.nodes = create_factory_nodes(read_max_factory_route(), self.images)
    self.product_to_recipes, self.ingredient_to_recipes, self.recipe_ingredient_to_amount = get_product_to_recipe_map(ReadSatisfactoryData())
    print(self.product_to_recipes)
    print(self.ingredient_to_recipes)
    print(self.recipe_ingredient_to_amount)

    self.frame_buffer = pygame.Surface(self.window_surface.get_size())
    self.frame_buffer.fill(Color(0, 0, 0, 255))
    self.alpha_frame_buffer = pygame.Surface(self.window_surface.get_size(), pygame.SRCALPHA)
    self.alpha_frame_buffer.fill(Color(0, 0, 0, 0))

    self.ui_manager = pygame_gui.UIManager(SCREEN_SIZE)

    product_label = pygame_gui.elements.UILabel(
      relative_rect=pygame.Rect((map_edge + 10, 10), (-1, 40)),
      text='Product Lookup:',
      manager=self.ui_manager,
    )

    self.product_search_box = pygame_gui.elements.UITextEntryLine(
      relative_rect=pygame.Rect((product_label.relative_rect.x + product_label.relative_rect.size[0] + 10, 10), (300, 40)),
      manager=self.ui_manager
    )

    rect = self.product_search_box.relative_rect
    self.item_rate_label = pygame_gui.elements.UILabel(
      relative_rect=pygame.Rect((rect.x + rect.size[0] + 10, 10), (-1, 40)),
      text='',
      manager=self.ui_manager,
    )

    rect = self.item_rate_label.relative_rect
    self.backward_button = pygame_gui.elements.UIButton(
      relative_rect=pygame.Rect((rect.x + rect.size[0] + 300, 10), (30, 40)),
      text='<',
      manager=self.ui_manager
    )

    rect = self.backward_button.relative_rect
    self.forward_button = pygame_gui.elements.UIButton(
      relative_rect=pygame.Rect((rect.x + rect.size[0] + 10, 10), (30, 40)),
      text='>',
      manager=self.ui_manager
    )

    html_text = (
      "<b>Welcome to the Game!</b><br><br>"
      "This is a standard <i>UITextBox</i> element. It automatically wraps text "
      "when it reaches the edge, and it will spawn a vertical scrollbar if "
      "there is too much content to fit inside the given height.<br><br>"
      "You can even change colors inline using <font color='#ff5555'>HTML tags</font>!"
    )

    self.node_info_box = pygame_gui.elements.UITextBox(
      html_text=html_text,
      relative_rect=pygame.Rect((map_edge + 10, 10 + 40 + 10), (700, 800)),  # (x, y), (width, height)
      manager=self.ui_manager,
    )

    self.clock = pygame.time.Clock()


  def mainLoop(self):
    while self.running:
      time_delta = self.clock.tick(60) / 1000.0
      # event handling, gets all event from the event queue
      for event in pygame.event.get():
        self.process_events(event)

      self.ui_manager.update(time_delta)
      self.update(time_delta)

      self.window_surface.blit(self.frame_buffer, (0, 0))
      # window_surface.blit(alpha_background, (0, 0))
      self.frame_buffer.fill(Color(0, 0, 0, 255))
      self.frame_buffer.blit(self.images['map'], (0, 0))
      self.ui_manager.draw_ui(self.window_surface)

      pygame.display.update()


  def update(self, time_delta):
    if self.run_simulation:
      for node in self.nodes:
        if node.locked:
          continue
        force = Vector2(0, 0)
        for edge in node.inputs:
          force += -.00001 * edge.rate * (node.position - self.nodes[edge.id].position)
        for edge in node.outputs:
          force += -.00001 * edge.rate * (node.position - self.nodes[edge.id].position)
        force -= 1 * node.velocity
        node.velocity += force * time_delta
        node.position += node.velocity * time_delta

      for node in self.nodes[524:]:
        node.update_dominant_force(self.nodes)

      # vor = Voronoi([[n.position.x, n.position.y] for n in self.nodes[524:]])
      # self.voronoi_points = vor.vertices

    for voronoi_point in self.voronoi_points:
      pygame.draw.circle(self.frame_buffer, Color(255, 0, 255, 255), Vector2(voronoi_point[0], voronoi_point[1]), 4)

    node_radius = 16
    interior_ratio = .93
    image_ratio = .8
    for node in self.nodes:
      if not node.visible:
        continue
      for edge in node.outputs:
        if not self.nodes[edge.id].visible:
          continue
        rect = pygame.draw.line(self.alpha_frame_buffer, edge.color, node.position, self.nodes[edge.id].position, 2)
        self.frame_buffer.blit(self.alpha_frame_buffer, rect, rect)
        self.alpha_frame_buffer.fill(Color(0, 0, 0, 0), rect)

    for i, node in enumerate(self.nodes):
      if not node.visible:
        continue

      if node.image is None:
        pygame.draw.circle(self.frame_buffer, node.rim_color, node.position, node_radius)
        pygame.draw.circle(self.frame_buffer, node.color, node.position, interior_ratio * node_radius)
        text_surface = self.font.render(str(i), True, (0, 0, 0))
        # 5. Draw (blit) the text surface onto the screen at (X, Y) coordinates
        self.frame_buffer.blit(text_surface, node.position - pygame.Vector2(text_surface.size) / 2)
      else:
        pygame.draw.circle(self.frame_buffer, node.rim_color, node.position, node_radius)
        pygame.draw.circle(self.frame_buffer, node.color, node.position, interior_ratio * node_radius)
        self.frame_buffer.blit(node.image, node.position.elementwise() - node.image.size[0] / 2)


    for node in self.nodes[524:]:
      pygame.draw.line(self.frame_buffer, Color(0, 0, 0, 255), node.position - node.dominant_force * 100, node.position + node.dominant_force * 100, 2)


  def update_node_info_box(self):
    if self.current_node is None or self.product_search_box.text not in self.product_to_recipes:
      self.node_info_box.set_text('')
      return

    lines = []
    if self.product_search_box.text in self.current_node.products:
      lines.append(f'<b>{self.product_search_box.text}</b>: {self.current_node.products[self.product_search_box.text]}/min')

    product_recipes = self.product_to_recipes[self.product_search_box.text]
    ingredient_recipes = self.ingredient_to_recipes.get(self.product_search_box.text, [])
    relevant_node_ids = set()
    for recipe in product_recipes:
      lines.append(f'<b>{recipe}</b>')
      for input in self.current_node.inputs:
        if recipe == input.recipe:
          node_id = input.id
          if self.current_node.id == node_id:
            node_id = 'self'
          else:
            relevant_node_ids.add(input.id)
          lines.append(f'&emsp;from {node_id}: {input.item} @ {input.rate}/min')
    lines.append('===========================')
    print(self.current_node.outputs)
    for recipe in ingredient_recipes:
      lines.append(f'<b>{recipe}</b>')
      for output in self.current_node.outputs:
        if recipe == output.recipe and output.item == self.product_search_box.text:
          node_id = output.id
          if self.current_node.id == node_id:
            node_id = 'self'
          else:
            relevant_node_ids.add(output.id)
          num_buildings = 0
          if f'{recipe}:{self.product_search_box.text}' in self.recipe_ingredient_to_amount:
            num_buildings = output.rate / (2.5 * self.recipe_ingredient_to_amount[f'{recipe}:{self.product_search_box.text}'])
          lines.append(f'&emsp;to {node_id}: {output.item} @ {output.rate}/min ({num_buildings} buildings)')

    self.node_info_box.set_text('<br>'.join(lines))

    for node in self.nodes:
      if node.id in relevant_node_ids:
        node.visible = True
        node.rim_color = Color(0, 0, 255, 255)
      elif node.id != self.current_node.id:
        node.visible = self.product_search_box.text in node.products
        node.rim_color = Color(255, 255, 255, 255)


  def process_events(self, event):
    self.ui_manager.process_events(event)

    if event.type == pygame.QUIT:
      self.running = False
      return

    if event.type == pygame_gui.UI_BUTTON_PRESSED:
      if event.ui_element == self.backward_button:
        self.handle_backwards()
      if event.ui_element == self.forward_button:
        self.handle_forwards()

    if event.type == pygame.MOUSEBUTTONUP:
      self.handle_mouse(*pygame.mouse.get_pos())

    if not self.product_search_box.is_focused and event.type == pygame.KEYDOWN:
      self.handle_key(event.key)

    if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
      if event.ui_element == self.product_search_box:
        self.handle_item_lookup(event.text)


  def handle_key(self, key):
    if key == pygame.K_SPACE:
      self.run_simulation = not self.run_simulation
      print(f'run_simulation: {self.run_simulation}')


  def handle_mouse(self, x, y):
    #print(f'(x, y): {(x, y)}, {screen_to_world(Vector2(x, y))})')
    world_pos = screen_to_world(Vector2(x, y))
    print(f'Vector2({world_pos.x}, {world_pos.y}),')
    # check click in reverse order to match display order
    for i in range(len(self.nodes) - 1, -1, -1):
      node = self.nodes[i]
      if not node.visible:
        continue
      if (node.position - Vector2(x, y)).magnitude() < 16:
        if self.current_node is not None and node.id == self.current_node.id:
          self.current_node.rim_color = Color(255, 255, 255, 255)
          self.current_node = None
          self.update_node_info_box()
          return

        print(i)
        # print(node)

        if self.product_search_box.text in node.products:
          print(f'{self.product_search_box.text}: {node.products[self.product_search_box.text]}')

        # for item_input in node.inputs:
        #   print(f'I: {item_input}')
        # for output in node.outputs:
        #   print(f'O: {output}')
        if self.current_node is not None:
          self.current_node.rim_color = Color(255, 255, 255, 255)
        self.current_node = node
        self.current_node.rim_color = Color(0, 255, 0, 255)
        self.update_node_info_box()
        return

  def handle_backwards(self):
    if self.history_index <= 0:
      return
    self.history_index -= 1
    item = self.search_history[self.history_index]
    self.product_search_box.set_text(item)
    self.handle_item_lookup(item, overwrite_history=False)

  def handle_forwards(self):
    if self.history_index == len(self.search_history) - 1:
      return
    self.history_index += 1
    item = self.search_history[self.history_index]
    self.product_search_box.set_text(item)
    self.handle_item_lookup(item, overwrite_history=False)

  def handle_item_lookup(self, item, overwrite_history=True):
    if item == '':
      for node in self.nodes:
        node.visible = True
        # node.rim_color = Color(255, 255, 255, 255)
      self.item_rate_label.set_text('')
    else:
      if overwrite_history:
        self.history_index += 1
        if self.history_index == len(self.search_history):
          self.search_history.append(item)
        else:
          self.search_history[self.history_index] = item

      total_rate = 0
      for node in self.nodes:
        if item not in node.products:
          node.visible = False
        else:
          node.visible = True
          # node.rim_color = Color(255, 255, 255, 255)
          total_rate += node.products[item]
      self.item_rate_label.set_text(f'total: {total_rate}/min')
    self.update_node_info_box()


  def cleanup(self):
    pass

def main():
  factory_explorer = FactoryExplorer()
  factory_explorer.run()

if __name__ == '__main__':
  main()