from typing import Any

import pygame
import pygame_gui

import json
import random

import numpy as np
from sympy import Matrix
from scipy import linalg
from scipy.optimize import linprog

from dataclasses import dataclass, field
from enum import Enum

Vector2 = pygame.math.Vector2

class EncodingState(Enum):
  INPUT = 0
  OUTPUT = 1

@dataclass
class FactoryNode:
  id: int
  recipeName: str
  inputRate: dict[str, float]
  outputRate: dict[str, float]
  scale_factor: float
  position: pygame.math.Vector2 = field(default_factory=lambda: Vector2(0, 0))

  def encode(self, items, encodingState: EncodingState):
    encoded_node = [0] * len(items)
    for i, item in enumerate(items):
      #TODO: consider separating inputs & outputs into separate columns
      if encodingState == EncodingState.INPUT and item in self.inputRate:
        encoded_node[i] += self.inputRate[item]
      if encodingState == EncodingState.OUTPUT and item in self.outputRate:
        encoded_node[i] += self.outputRate[item]
    return encoded_node

  def encodeInput(self, items):
    encoded_node = [0] * len(items)
    for i, item in enumerate(items):
      if item in self.inputRate:
        encoded_node[i] += self.inputRate[item]
    return encoded_node

  def encodeOutput(self, items):
    encoded_node = []
    outputItems = sorted(list(self.outputRate.keys()))
    if len(outputItems) > 1:
      print(f'{self.recipeName}: {outputItems}')
    for outputItem in outputItems:
      encoded_node.append([0] * len(items))
      for i, item in enumerate(items):
        if item == outputItem:
          encoded_node[-1][i] += self.outputRate[item]
    return encoded_node

def extractRREFSolution(rref, shape):
  solution = np.zeros(shape)
  j = 0
  for i in range(rref.shape[0]):
    while rref[i, j] != 1:
      j += 1
      if j == rref.shape[1] - shape[1]:
        return solution
    solution[i, 0] = rref[i, -1]
    j += 1
    if j == rref.shape[1] - shape[1]:
      return solution
  return solution

def ReadNodeData():
  # Open and read the JSON file
  with open('node_locations.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)

def ReadFactoryData():
  # Open and read the JSON file
  with open('max_factory.json', 'r', encoding="utf8") as f:
    jsonStr = f.read()
  return json.loads(jsonStr)

# def null_space(A, rcond=None):
#   u, s, vh = np.linalg.svd(A, full_matrices=True)
#   M, N = u.shape[0], vh.shape[1]
#   if rcond is None:
#     rcond = np.finfo(s.dtype).eps * max(M, N)
#   tol = np.amax(s) * rcond
#   num = np.sum(s > tol, dtype=int)
#   Q = vh[num:, :].T.conj()
#   return Q

def applySumConstrains(solution, node_null_space):
  # apply alpha constraint
  ones = np.ones((solution.shape[1], 1))
  kron_matrix = np.kron(ones.T, node_null_space)

  sum_constraint_y = np.ones((solution.shape[0], 1)) - np.matmul(solution, ones)
  # sum_constraint_y = np.zeros_like(sum_constraint_y)

  # rref_M = Matrix(np.column_stack((kron_matrix, sum_constraint_y))).rref()[0]
  # print('reduced row echelon')
  # rref = np.array(rref_M).astype(np.float64)
  # print(rref.shape)
  # print(rref)

  # solution_info2 = linalg.lstsq(kron_matrix, sum_constraint_y)
  # solution2 = solution_info2[0]
  solution2 = np.matmul(linalg.pinv(kron_matrix), sum_constraint_y)
  #solution2 = extractRREFSolution(rref, (kron_matrix.shape[1], sum_constraint_y.shape[1]))

  kron_null_space = linalg.null_space(kron_matrix)
  #kron_null_space = null_space(kron_matrix)

  free_var2_shape = (kron_null_space.shape[1], 1)
  free_vars2 = np.random.rand(*free_var2_shape)

  print('kron_matrix')
  print(kron_matrix.shape)
  print(kron_matrix)

  print('sum_constraint_y')
  print(sum_constraint_y.shape)
  print(sum_constraint_y)

  print('solution2')
  print(solution2.shape)
  print(solution2)

  print('kron_null_space')
  print(kron_null_space.shape)
  print(kron_null_space)

  print('free_vars2.shape')
  print(free_var2_shape)

  print('validate solution2')
  print(np.matmul(kron_matrix, solution2) - sum_constraint_y)
  print('validate solution2 with free vars')
  print(np.matmul(kron_null_space, free_vars2))
  print(np.matmul(kron_matrix, solution2 + np.matmul(kron_null_space, free_vars2)) - sum_constraint_y)

  return solution2, kron_null_space, ones, free_var2_shape


def getBaseAlphaMatrix2(scale_matrix, node_matrix, recipe_matrix):
  np.set_printoptions(linewidth=2000, suppress=True)
  print('recipe_matrix')
  print(recipe_matrix.shape)
  print(recipe_matrix)
  for i in range(0, recipe_matrix.shape[0]):
    print(recipe_matrix[i, :])

  print('node_matrix')
  print(node_matrix.shape)
  print(node_matrix)

  print('scale_matrix')
  print(scale_matrix.shape)
  print(scale_matrix)

  scaled_node_matrix = np.matmul(node_matrix, scale_matrix)
  #rref = Matrix(np.column_stack((scaled_node_matrix, recipe_matrix))).rref()[0]
  # solution_info = linalg.lstsq(scaled_node_matrix, recipe_matrix)
  # solution = solution_info[0]
  solution = np.matmul(linalg.pinv(scaled_node_matrix), recipe_matrix)

  #print(np.array(rref).astype(np.float64))

  node_null_space = linalg.null_space(scaled_node_matrix)

  free_var_shape = (node_null_space.shape[1], solution.shape[1])
  free_vars = np.random.rand(*free_var_shape)

  print('solution')
  print(solution.shape)
  print(solution)

  print('node_null_space')
  print(node_null_space.shape)
  print(node_null_space)

  print('free_var_shape')
  print(free_var_shape)

  print('verify solution')
  print(np.matmul(scaled_node_matrix, solution))
  print('verify solution with free vars')
  print(np.matmul(scaled_node_matrix, solution + np.matmul(node_null_space, free_vars)))

  solution2, kron_null_space, ones, free_var2_shape = applySumConstrains(solution, node_null_space)
  solution2 = np.reshape(solution2, free_var_shape, order='F')
  alpha = solution + np.matmul(node_null_space, solution2)
  print(alpha)
  for i in range(0, alpha.shape[0]):
    print(alpha[i, :])
  print('check sum constraint')
  alpha = solution + np.matmul(node_null_space, solution2)
  print(np.matmul(alpha, ones))
  print('verify solution')
  print(np.matmul(scaled_node_matrix, alpha))
  print('check sum constraint with free vars')
  free_vars2 = np.random.rand(*free_var2_shape)
  free_vars = solution2 + np.reshape(np.matmul(kron_null_space, free_vars2), free_var_shape, order='F')
  solution = solution + np.matmul(node_null_space, free_vars)
  print(np.matmul(solution, ones))
  print('verify solution')
  print(np.matmul(scaled_node_matrix, solution))
  return 0


def getBaseAlphaMatrix(scale_matrix, node_matrix, recipe_matrix):
  np.set_printoptions(linewidth=2000, suppress=True)
  print('recipe_matrix')
  print(recipe_matrix.shape)
  print(recipe_matrix)

  print('node_matrix')
  print(node_matrix.shape)
  print(node_matrix)

  node_kron_matrix = np.kron(np.identity(recipe_matrix.shape[1]), node_matrix)
  print('node_kron_matrix')
  print(node_kron_matrix.shape)
  #print(np.linalg.matrix_rank(node_kron_matrix))

  recipe_vector = recipe_matrix.flatten(order='F')

  alpha_mask = np.matmul(node_matrix.transpose(), recipe_matrix)
  alpha_mask_vector = alpha_mask.flatten(order='F')

  print('alpha_mask')
  print(alpha_mask.shape)
  print(alpha_mask_vector.shape)
  print(alpha_mask)

  node_kron_matrix_known = node_kron_matrix[:, np.where(alpha_mask_vector == 0)[0]]
  print('node_kron_matrix_known')
  print(node_kron_matrix_known.shape)
  print(node_kron_matrix_known)

  node_kron_matrix_unknown = node_kron_matrix[:, np.where(alpha_mask_vector != 0)[0]]
  print('node_kron_matrix_unknown')
  print(node_kron_matrix_unknown.shape)
  print(node_kron_matrix_unknown)

  recipe_vector_prime = recipe_vector - np.matmul(node_kron_matrix_known, alpha_mask_vector[alpha_mask_vector == 0])
  print('recipe_vector_prime')
  print(recipe_vector_prime.shape)
  print(recipe_vector_prime)

  #alpha_unknown = np.matmul(linalg.pinv(node_kron_matrix_unknown), recipe_vector_prime)
  results = linalg.lstsq(node_kron_matrix_unknown, recipe_vector_prime)
  alpha_unknown = results[0]
  # alpha_mask_unknown = recipe_vector_prime / node_kron_matrix_known
  print('alpha_mask_unknown')
  print(alpha_unknown.shape)
  print(alpha_unknown)
  print(results[2])

  alpha_vector = np.zeros_like(alpha_mask_vector)
  alpha_vector[alpha_mask_vector != 0] = alpha_unknown

  alpha = np.reshape(alpha_vector, alpha_mask.shape, order='F')
  print('alpha')
  print(alpha.shape)
  print(alpha)
  for i in range(0, alpha.shape[0]):
    print(alpha[i, :])

  node_kron_matrix_unknown_null = linalg.null_space(node_kron_matrix_unknown)
  print('node_kron_matrix_unknown_null')
  print(node_kron_matrix_unknown_null.shape)
  print(node_kron_matrix_unknown_null)

  free_var_shape = (node_kron_matrix_unknown_null.shape[1],)
  free_vars = np.random.rand(*free_var_shape)

  print('free_var_shape')
  print(free_var_shape)
  print(alpha_vector[alpha_mask_vector != 0].shape)

  alpha_vector2 = np.zeros_like(alpha_mask_vector)
  alpha_vector2[alpha_mask_vector != 0] = alpha_unknown + np.matmul(node_kron_matrix_unknown_null, free_vars)
  alpha2 = np.reshape(alpha_vector2, alpha_mask.shape, order='F')
  print('alpha2')
  for i in range(0, alpha2.shape[0]):
    print(alpha2[i, :])

  row_sum_matrix = np.zeros((alpha_mask.shape[0], alpha_mask_vector.shape[0]))
  for i in range(row_sum_matrix.shape[0]):
    for j in range(alpha_mask.shape[1]):
      row_sum_matrix[i, j * row_sum_matrix.shape[0] + i] = 1

  print('row_sum_matrix')
  print(row_sum_matrix.shape)
  print(row_sum_matrix)

  row_sum = np.matmul(row_sum_matrix, alpha_vector)
  print('row_sum')
  print(row_sum.shape)
  print(row_sum)

  row_sum_unknown = np.matmul(row_sum_matrix[:, np.where(alpha_mask_vector != 0)[0]], alpha_unknown)
  print('row_sum_unknown')
  print(row_sum_unknown.shape)
  print(row_sum_unknown)

  row_sum_unknown_matrix = row_sum_matrix[:, np.where(alpha_mask_vector != 0)[0]]
  # np.matmul(row_sum_unknown_matrix, alpha_unknown + np.matmul(node_kron_matrix_unknown_null, free_vars)) = np.ones((alpha_mask.shape[0],))
  # np.matmul(row_sum_unknown_matrix, alpha_unknown) + np.matmul(row_sum_unknown_matrix, np.matmul(node_kron_matrix_unknown_null, free_vars)) = np.ones((alpha_mask.shape[0],))
  # np.matmul(np.matmul(row_sum_unknown_matrix, node_kron_matrix_unknown_null), free_vars)) = np.ones((alpha_mask.shape[0],)) - np.matmul(row_sum_unknown_matrix, alpha_unknown)
  sum_constraint = np.ones((alpha_mask.shape[0],)) - np.matmul(row_sum_unknown_matrix, alpha_unknown)
  A = np.matmul(row_sum_unknown_matrix, node_kron_matrix_unknown_null)
  #solution = np.matmul(linalg.pinv(A), sum_constraint)
  results = linalg.lstsq(A, sum_constraint)
  solution = results[0]

  print('solution')
  print(results[2])

  constrained_alpha_unknown = alpha_unknown + np.matmul(node_kron_matrix_unknown_null, solution)
  row_sum_unknown2 = np.matmul(row_sum_matrix[:, np.where(alpha_mask_vector != 0)[0]], constrained_alpha_unknown)
  print('row_sum_unknown2')
  print(row_sum_unknown2.shape)
  print(row_sum_unknown2)

  constrained_alpha_vector = np.zeros_like(alpha_mask_vector)
  constrained_alpha_vector[alpha_mask_vector != 0] = constrained_alpha_unknown
  constrained_alpha = np.reshape(constrained_alpha_vector, alpha_mask.shape, order='F')
  print('constrained_alpha')
  for i in range(0, constrained_alpha.shape[0]):
    print(constrained_alpha[i, :])

  A_null = linalg.null_space(A)
  print('A_null')
  print(A_null.shape)
  print(A_null)

  free_var_shape2 = (A_null.shape[1],)
  free_vars2 = np.random.rand(*free_var_shape2)

  print('free_var_shape2')
  print(free_var_shape2)

  # alpha_unknown + np.matmul(node_kron_matrix_unknown_null, solution + np.matmul(A_null, free_vars2)) > np.zeros_like(alpha_unknown)
  # np.matmul(node_kron_matrix_unknown_null, solution + np.matmul(A_null, free_vars2)) > np.zeros_like(alpha_unknown) - alpha_unknown
  # np.matmul(node_kron_matrix_unknown_null, solution) + np.matmul(node_kron_matrix_unknown_null, np.matmul(A_null, free_vars2))) > np.zeros_like(alpha_unknown) - alpha_unknown
  # np.matmul(np.matmul(node_kron_matrix_unknown_null, A_null), free_vars2)) > np.zeros_like(alpha_unknown) - alpha_unknown - np.matmul(node_kron_matrix_unknown_null, solution)
  B = np.matmul(node_kron_matrix_unknown_null, A_null)
  positive_constraint = np.zeros_like(alpha_unknown) - alpha_unknown - np.matmul(node_kron_matrix_unknown_null, solution)
  #positive_constraint = .01*np.ones_like(alpha_unknown) - alpha_unknown - np.matmul(node_kron_matrix_unknown_null, solution)

  print('B')
  print(B.shape)
  print(np.count_nonzero(B))
  # for i in range(0, B.shape[0]):
  #   print(B[i, :])

  print('positive_constraint')
  print(positive_constraint.shape)
  print(positive_constraint[:200])

  print('constraint test')
  print(np.matmul(B, free_vars2)[:200])

  zero_rows = np.nonzero(np.all(np.abs(B) >= 1e-7 * np.ones_like(B), axis=1))
  print('zero_rows')
  print(zero_rows[0].shape)
  print(zero_rows[0])
  corner = B[zero_rows[0][0], :]
  print('corner')
  print(corner)
  # opt = linprog(c=corner, A_ub=-B, b_ub=-positive_constraint, bounds=(None, None), method='highs-ipm') # options={'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8}
  opt = linprog(c=node_kron_matrix_unknown_null[0,:], A_ub=-node_kron_matrix_unknown_null, b_ub=-alpha_unknown, bounds=(None, None), method='highs-ipm')
  print(opt)
  print(opt.x)
  print(len(opt.x))

  # constrained_alpha_vector2 = np.zeros_like(alpha_mask_vector)
  # constrained_alpha_unknown_vector2 = alpha_unknown + np.matmul(node_kron_matrix_unknown_null, solution + np.matmul(A_null, free_vars2))
  # constrained_alpha_vector2[alpha_mask_vector != 0] = constrained_alpha_unknown_vector2
  # constrained_alpha2 = np.reshape(constrained_alpha_vector2, alpha_mask.shape, order='F')
  # print('constrained_alpha_unknown_vector2')
  # print(constrained_alpha_unknown_vector2[:200])
  #
  # zero_rows = np.all(np.abs(B) >= 1e-8 * np.ones_like(B), axis=1)
  # zero_columns = np.all(np.abs(B) >= 1e-8 * np.ones_like(B), axis=0)
  # print('zero_rows')
  # print(zero_rows.shape)
  # print(np.count_nonzero(zero_rows))
  # print('zero_columns')
  # print(zero_columns.shape)
  # print(np.count_nonzero(zero_columns))
  #
  # B2 = B[zero_rows, :]
  # print(B2.shape)
  #
  # rows = np.arange(B.shape[1], dtype=int)
  # #rows[-1] += 2
  # for n in range(B2.shape[0]-B.shape[1]):
  #   print(n)
  #   rows2 = rows + n
  #   try:
  #     point_candidate = np.matmul(linalg.inv(B2[rows2, :]), positive_constraint[rows2])
  #     point_candidate_test2 = np.abs(np.matmul(B2[rows2, :], point_candidate) - positive_constraint[rows2]) < (1e3 * np.ones_like(
  #       positive_constraint[rows2]))
  #     point_candidate_test = np.matmul(B, point_candidate) - positive_constraint >= -.001 * np.ones_like(positive_constraint)
  #     print(f'passing inequalities: {np.count_nonzero(point_candidate_test)}')
  #     print(f'passing inequalities2: {np.count_nonzero(point_candidate_test2)}')
  #     if np.all(point_candidate_test):
  #       break
  #   except linalg.LinAlgError:
  #     pass
  #
  #
  #
  constrained_alpha_vector3 = np.zeros_like(alpha_mask_vector)
  # constrained_alpha_unknown_vector3 = alpha_unknown + np.matmul(node_kron_matrix_unknown_null,
  #                                                               solution + np.matmul(A_null, opt.x))
  constrained_alpha_unknown_vector3 = alpha_unknown + np.matmul(node_kron_matrix_unknown_null,  opt.x)
  print('constrained_alpha_unknown_vector3')
  print(constrained_alpha_unknown_vector3[:200])

  # for i in range(0, constrained_alpha2.shape[0]):
  #   print(constrained_alpha2[i, :])
  return 0

# def getBaseAlphaMatrix(scale_matrix, node_matrix, recipe_matrix):
#   np.set_printoptions(linewidth=2000, suppress=True)
#   print('recipe_matrix')
#   print(recipe_matrix.shape)
#   print(recipe_matrix)
#
#   print('node_matrix')
#   print(node_matrix.shape)
#   print(node_matrix)
#
#   print('scale_matrix')
#   print(scale_matrix.shape)
#   print(scale_matrix)
#   opt = optimize.linprog(c=encodedGoal, A_eq=encodedRecipes, b_eq=encodedLimits, bounds=bounds, method='highs')

# 0.          0.          0.          0.
# 0.          0.         -0.02431913  0.
# 1.          0.          0.          0.
# 0.          1.          0.01201495  0.

# [[14095.89766036     0.             0.         55204.10233964]
#  [ 8457.53859622  4968.1775112     74.28389258     0.        ]
#  [19734.2567245      0.             0.             0.        ]
#  [    0.          3864.13806427    46.42743286     0.        ]]


def main():
  resources = ReadNodeData()
  max_factory = ReadFactoryData()

  print(max_factory)

  item_set = set()
  for recipe in max_factory:
    for ingredient in max_factory[recipe]['ingredients']:
      item_set.add(ingredient)
    for product in max_factory[recipe]['products']:
      item_set.add(product)

  print(item_set)
  items = sorted(list(item_set))
  print(items)

  node_id = 0
  factories = []
  for item in items:
    if item in resources:
      for node_info in resources[item]:
        factories.append(FactoryNode(node_id, item, {}, {item: node_info['rate']}, 1, Vector2(node_info['x'], node_info['y'])))
        node_id += 1

  recipes = []
  for factory in max_factory:
    if 'Desc_Water_C' in max_factory[factory]['ingredients']:
      del max_factory[factory]['ingredients']['Desc_Water_C']
    if 'Desc_Water_C' in max_factory[factory]['products']:
      del max_factory[factory]['products']['Desc_Water_C']

    for product in max_factory[factory]['products']:
      factories.append(
        FactoryNode(node_id, factory, max_factory[factory]['ingredients'], {product: max_factory[factory]['products'][product]}, 1))
      node_id += 1
    recipes.append(
      FactoryNode(-1, factory, max_factory[factory]['ingredients'], max_factory[factory]['products'], 1))

  recipe_matrix = []
  for recipe in recipes:
    recipe_matrix.append(recipe.encode(items, EncodingState.INPUT))
    #recipe_matrix.append(recipe.encodeInput(items))
  recipe_matrix = np.array(recipe_matrix).transpose()

  node_matrix = []
  for factory in factories:
    node_matrix.append(factory.encode(items, EncodingState.OUTPUT))
    #node_matrix.extend(factory.encodeOutput(items))
  node_matrix = np.array(node_matrix).transpose()

  scale_matrix = np.diag([factory.scale_factor for factory in factories])

  # solve
  print(getBaseAlphaMatrix(scale_matrix, node_matrix, recipe_matrix))

  for factory in factories:
    print(factory)
  print(items)

if __name__ == '__main__':
  main()