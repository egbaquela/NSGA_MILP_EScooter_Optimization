import sys
import json
import time
import math
import random

from ortools.linear_solver import pywraplp
import numpy as np
from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation
from pymoo.visualization.scatter import Scatter


hist = {}
def solve_milp(max_docks_per_station, 
               cant_total_vehiculos, 
               demanda, 
               min_docks_por_zona, 
               max_docks_por_zona, 
               max_veh_por_zona, 
               max_cap_relocacion,
               zs = [],
               z1_weight = 1.0):
    
    hist_key = "0"
    if len(zs)>0:
        hist_key = '-'.join([str(int(x)) for x in zs])
    if (len(zs) > 0) and (hist_key in hist):
        return hist[hist_key]
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    infinity = solver.infinity()
    
    # Variables
    x = {} # x_ijt proporcion de demanda cubierta desde i a j en tiempo t
    for i in range(cant_zonas):
        aux_x_i = {}
        for j in range(cant_zonas):
            aux_x_i_j = {}
            for t in range(cant_periodos):
                aux_x_i_j[t] = solver.NumVar(0, 1, 'x[%i,%i,%i]' % (i,j,t))
            aux_x_i[j] =aux_x_i_j
        x[i] = aux_x_i

    v = {} # v_it vehículos al inicio
    for i in range(cant_zonas):
        aux_v_i = {}
        for t in range(cant_periodos):
             aux_v_i[t] = solver.NumVar(0, max_veh_por_zona[i], 'v[%i,%i]' % (i,t))
        v[i] = aux_v_i

    r = {} # r_ijt vehículos relocados
    for i in range(cant_zonas):
        aux_r_i = {}
        for j in range(cant_zonas):
            aux_r_i_j = {}
            for t in range(cant_periodos):
                aux_r_i_j[t] = solver.NumVar(0, max_cap_relocacion[i], 'r[%i,%i,%i]' % (i,j,t))
            aux_r_i[j] =aux_r_i_j
        r[i] = aux_r_i

    y = {} # y_i 1 si la zona tiene estación
    for i in range(cant_zonas):
        y[i] = solver.IntVar(0, 1, 'y[%i]' % (i))

    z = {} # z_i cant docks
    for i in range(cant_zonas):
        z[i] = solver.NumVar(0, max_docks_por_zona[i], 'z[%i]' % (i))
    
    w_cant_veh = solver.IntVar(0, cant_total_vehiculos, 'y[%i]' % (i))
 
    # Función objetivo
    objective = solver.Objective()
    for i in range(cant_zonas):
        for j in range(cant_zonas):
            for t in range(cant_periodos):
                objective.SetCoefficient(x[i][j][t], z1_weight * demanda[i][j][t])
    for i in range(cant_zonas):
        objective.SetCoefficient(z[i], (-1)*(1-z1_weight))

    objective.SetMaximization()
    
    # Restricciones

    # Restricción 1
    for i in range(cant_zonas):
        for t in range(1, cant_periodos):
            constraint = solver.RowConstraint(0, 0, 'Balance de flujo entre v[%i][%i] y v[%i][%i]' % (i, t, i, t-1))
            constraint.SetCoefficient(v[i][t], -1)
            constraint.SetCoefficient(v[i][t-1], 1)
            for j in range(cant_zonas):
                constraint.SetCoefficient(x[i][j][t-1], -1*demanda[i][j][t-1])
                constraint.SetCoefficient(x[j][i][t-1], demanda[j][i][t-1])
                constraint.SetCoefficient(r[j][i][t-1], 1)
                constraint.SetCoefficient(r[i][j][t-1], -1)

    # Restricción 2
    for i in range(cant_zonas):
        constraint = solver.RowConstraint(0, 0, 'v[%i][0] = v[%i][cant_periodos-1]' % (i, i))
        constraint.SetCoefficient(v[i][0], 1)
        constraint.SetCoefficient(v[i][cant_periodos-1], -1)

    # Restricción 3
    for i in range(cant_zonas):
        constraint = solver.RowConstraint(-max_docks_por_zona[i], 0, 'z[%i] <= z_max * y[%i]' % (i, i))
        constraint.SetCoefficient(z[i], 1)
        constraint.SetCoefficient(y[i], -1*max_docks_por_zona[i])

    # Restricción 4
    for i in range(cant_zonas):
        constraint = solver.RowConstraint(0, max_docks_por_zona[i], 'z[%i] >= z_min * y[%i]' % (i, i))
        constraint.SetCoefficient(z[i], 1)
        constraint.SetCoefficient(y[i], -1)

    # Restricción 5
    for i in range(cant_zonas):
        for t in range(cant_periodos):
            constraint = solver.RowConstraint(0, infinity, 'Bicicletas en %i al inicio mayores o iguales a la demanda en el periodo %i' % (i, t))
            constraint.SetCoefficient(v[i][t], 1)
            for j in range(cant_zonas):
                constraint.SetCoefficient(x[i][j][t], -1*demanda[i][j][t])

    # Restricción 6
    for i in range(cant_zonas):
        for t in range(cant_periodos):
            constraint = solver.RowConstraint(-infinity, 0, 'Link entre v[%i][%i] y z[%i] - Parte 01' % (i, t, i))
            constraint.SetCoefficient(v[i][t], 1)
            constraint.SetCoefficient(z[i], -1)

    # Restricción 7
    for i in range(cant_zonas):
        for t in range(cant_periodos):
            constraint = solver.RowConstraint(0, infinity, 'Link entre v[%i][%i] y z[%i] - Parte 02' % (i, t, i))
            constraint.SetCoefficient(v[i][t], 1)
            constraint.SetCoefficient(z[i], -0.25)

    # Restricción 8
    for i in range(cant_zonas):
        for t in range(cant_periodos):
            constraint = solver.RowConstraint(0, infinity, 'Bicicletas relocadas en la zona %i en el periodo %i tienen que se menores o iguales a las disponibles' % (i, t))
            constraint.SetCoefficient(v[i][t], -1)
            for j in range(cant_zonas):
                constraint.SetCoefficient(r[i][j][t], 1)          

    # Restricción 9
    for t in range(cant_periodos):
        constraint = solver.RowConstraint(0, 0, 'En cada instante %i, las bicicleta en el sistema deben ser iguales a la cantidad disponible' % (t))
        constraint.SetCoefficient(w_cant_veh, -1)
        for i in range(cant_zonas):
            constraint.SetCoefficient(v[i][t], 1)

    # Restricción 17
    for i in range(cant_zonas):
        for t in range(cant_periodos):
            constraint = solver.RowConstraint(0, 1, 'Proporción total igual a 1 para zona %i en periodo %i' % (i, t))
            for j in range(cant_zonas):
                constraint.SetCoefficient(x[i][j][t], 1)

    # Restricción 18            
    for i in range(cant_zonas):
        for j in range(cant_zonas):
            for t in range(cant_periodos):
                constraint = solver.RowConstraint(-1, 0, 'Vinculación de y[%i] con x[%i][%i][%i] - Parte 01' % (i, i, j, t))
                constraint.SetCoefficient(y[i], -1)
                constraint.SetCoefficient(x[i][j][t], 1) 

    # Restricción 19            
    for i in range(cant_zonas):
        for j in range(cant_zonas):
            for t in range(cant_periodos):
                constraint = solver.RowConstraint(-1, 0, 'Vinculación de y[%i] con x[%i][%i][%i] - Parte 02' % (i, i, j, t))
                constraint.SetCoefficient(y[j], -1)
                constraint.SetCoefficient(x[i][j][t], 1) 
    
    if len(zs)>0:
        for i in range(len(zs)):
            #value = 0
            #if ys[i]==True:
            #    value=1
            value = int(zs[i])
            constraint = solver.RowConstraint(value, value, 'Dar a z[%i] un valor fijo.' % (i))
            constraint.SetCoefficient(z[i], 1)
            if value > 0:
                value = 1
            constraint = solver.RowConstraint(value, value, 'Dar a y[%i] un valor fijo.' % (i))
            constraint.SetCoefficient(y[i], 1)

    mp_params = pywraplp.MPSolverParameters()
    solver.SetSolverSpecificParametersAsString("limits/stallnodes = 250")
    solver.SetSolverSpecificParametersAsString("misc/improvingsols = TRUE")
    mp_params.SCALING = mp_params.SCALING_ON

    status = solver.Solve()

    result = -1
    cant_estaciones = sum([y_var.value() for y_var in y])
    cant_docks = sum([z_var.value() for z_var in z])
    if status <= 1:
        result = solver.Objective().Value()
    if (len(zs) > 0):
        hist[hist_key] = (result, x, z, cant_estaciones, cant_docks)


    return result, x, z, cant_estaciones, cant_docks

class escooter_sharing_station_optimization(ElementwiseProblem):

    def __init__(self, n_var=1, n_obj=2, n_constr=0, xl=np.zeros(1), xu=np.ones(1)):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        aux = solve_milp(max_docks_per_station, 
               cant_total_vehiculos, 
               demanda, 
               min_docks_por_zona, 
               max_docks_por_zona, 
               max_veh_por_zona, 
               max_cap_relocacion,
               zs=x)
        #print(-aux, np.sum(x))
        out["F"] = (-aux[0], np.sum(x))
        #out["G"] = np.column_stack([0.1 - out["F"], out["F"] - 0.5])

###### Ejecución de ejemplo #####
# Seteo de escenario
hist = {}
# Indices
cant_zonas = 4
cant_periodos = 14

# Parametros
max_docks_per_station = 10
cant_total_vehiculos = 10  # Debe ser mayor que la suma de la demanda en todas las zonas en el periodo pico
                           # Debe ser menor o igual a la suma de todas las docks en todas las zonas

demanda = {} # u_ijt
demanda_total = 0
for i in range(cant_zonas):
    aux_dem_i = {}
    for j in range(cant_zonas):
        aux_dem_i_j = {}
        for t in range(cant_periodos):
            aux_dem_i_j[t] = 5
            demanda_total += 5
        aux_dem_i[j] =aux_dem_i_j
        
    demanda[i] = aux_dem_i 

min_docks_por_zona = [0 for i in range(cant_zonas)]  # mínimo condicionado a que se seleccione esa zona  
max_docks_por_zona = [10 for i in range(cant_zonas)]
max_veh_por_zona = [10 for i in range(cant_zonas)]
max_cap_relocacion = [10 for i in range(cant_zonas)]

# Optimización

problem = escooter_sharing_station_optimization(n_var=cant_zonas, 
                                                n_obj=2, 
                                                n_constr=0, 
                                                xl=np.zeros(cant_zonas), 
                                                xu=max_docks_per_station*np.ones(cant_zonas))

algorithm = NSGA2(pop_size=100,
                  sampling=get_sampling("int_random"),
                  crossover=get_crossover("int_two_point"),
                  mutation=get_mutation("int_pm"),
                  eliminate_duplicates=False)

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=1,
               verbose=True)


plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
