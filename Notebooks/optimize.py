#!/usr/bin/env python3
import sys


# Bibliotecas de uso general en el cuaderno
from pymoo.core.problem import Problem
import hashlib
import argparse

import numpy as np
from myutils import *
import pickle
import os
import signal

from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair

from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.indicators.hv import HV




class MinPMedian(ElementwiseProblem):

    def __init__(self, costs, cap, dem, assign_type = 'classic', max_cost = 9000):
        
        self.costs = costs
        self.num_bases, self.num_demand = costs.shape # Número de bases y colapsos
        self.capacities = cap
        self.demands = dem
        self.max_cost = max_cost
        self.map_to_bases = dict()
        self.assign_type = assign_type
        self.explored = set()
        self.classic_assign = dict()
        
        super().__init__(n_var=costs.shape[0], n_obj=2, n_ieq_constr=2, var_type=bool)
        
    def assign(self, sol, assign_type = None):

            costs = copy(self.costs)
            cap = copy(self.capacities)
            dem = copy(self.demands)
            z = copy(sol)
            # Indices de las bases marcadas con 1
            indices_bases = np.nonzero(z*1)[0]  
            
            # Si en algún momento se queda con menos de dos bases, se agregan nuevas aleatoriamente

            if len(indices_bases) < 2:
                # NUmero de bases a añadir aleatoriamente
                num_bases_to_add = np.random.randint(2, costs.shape[0]/2)
                
                # Mientras no se hayan agregado todas las bases
                while len(indices_bases) < num_bases_to_add:
                    
                    # Selecciona una base aleatoria
                    new_base = np.random.randint(0, costs.shape[0])
                    
                    # Si la base no está en la lista de bases, se agrega
                    if new_base not in indices_bases:
                        z[new_base] = 1
                        indices_bases = np.append(indices_bases, new_base)

            
            # Para cada punto de demanda (columna) ordenar las bases por costo,
            # es decir, mueve las filas, arriba las bases menos costosas por cada punto de demanda
            columnas = np.argsort(costs[indices_bases], axis = 0)

            # Obtener las bases más cercanas a cada colapso
            nearest_base = indices_bases[columnas]
            
            a = [-1 for _ in range(costs.shape[1])]
    
            
            if not assign_type:
                a_type = self.assign_type
            else:
                a_type = assign_type
            
            
            # Asignación clásica --------------------------------------------------------------------
            # ---------------------------------------------------------------------------------------
            if a_type == 'classic':
                
                new_indices = [i for i in range(costs.shape[1])]
                np.random.shuffle(new_indices)
                       
            
            
            # Asignación por urgencias --------------------------------------------------------------
            # ---------------------------------------------------------------------------------------
            elif a_type == 'urgencies':
        
                # Obtiene la matriz de costos ordenada por costo por cada punto de demanda
                # a diferencia de la otra matriz, no contiene el ìndice si no el valor
                nearest_base_value = np.sort(costs[indices_bases], axis=0)

                # Calculas las prioridades de los puntos de demanda
                priorities = nearest_base_value[1] - nearest_base_value[0]

                new_indices = np.argsort(priorities)[::-1]
               
            
            
            # Asignación por urgencias y capacidad ---------------------------------------------------
            # ---------------------------------------------------------------------------------------
            elif a_type == 'urgencies_cap':
                
                # Obtiene la matriz de costos ordenada por costo por cada punto de demanda
                # a diferencia de la otra matriz, no contiene el ìndice si no el valor
                nearest_base_value = np.sort(costs[indices_bases], axis=0)

                # Calculas las prioridades de los puntos de demanda
                priorities = nearest_base_value[1] - nearest_base_value[0]
                
                new_indices = np.argsort(dem)[::-1]
                
                # Ordena primero por demanda y luego por prioridad (urgencia)
                new_indices = np.lexsort((priorities, dem))[::-1]
                
                
                
            # Por cada punto de demanda, identifica la más cercana disponible
            for j in new_indices:
                
                # Recorre las bases (columnas)
                for i in range(nearest_base.shape[0]):
                    
                    # Si la base en la posición [j] tiene capacidad para atender el punto i, 
                    # se asigna y se descuenta del vector de capacidades
                    if cap[nearest_base[i][j]] >= dem[j]:
                        cap[nearest_base[i][j]] -= dem[j]
                        a[j] = nearest_base[i][j]
                        
                        # Se procede al siguiente punto de demanda
                        break
                    
                    # Si no hay capacidad, se procede a la siguiente base
                    else:
                        continue
            
            
            
            
            # Reparación de solución ----------------------------------------------------------------
            # ---------------------------------------------------------------------------------------
            
            bases_dif, counts = np.unique(a, return_counts=True)
            
            # Verifica que todos los puntos de demanda hayan sido asignados
            for point, base in enumerate(a):
                
                # Si para un punto de demanda no se asignó base, o sólo se asignó ese
                # punto a la base, se habilita la base más cercana disponible
                
                if base == -1 or counts[bases_dif == base][0] == 1:
                    
                    bases_to_point = costs[:,point]
                    nearest_base = np.argsort(bases_to_point)
                    
                    for b in nearest_base:
                        if cap[b] >= dem[point]:
                            cap[b] -= dem[point]
                            a[point] = b
                            z[b] = 1
                            break   
            
            # Vector de solución reparado:
            z = np.zeros(costs.shape[0])
            z[a] = 1
            
            # En caso de haber usado el método clásico, para asegurar congruencia entre vectores Z,
            # se guarda la asignación clásica en un diccionario, o se toma la que estaba
            if a_type == 'classic':
                hash_z = hashlib.sha256(z).hexdigest()
                
                if hash_z not in self.classic_assign:
                    self.classic_assign[hash_z] = a
                else:
                    a = self.classic_assign.get(hash_z)
            
            return a, z
            
    def costos_rutas(self, a):
        return [self.costs[i][j] for j,i in enumerate(a)]

    def f1(self, a):
        return  round(sum(self.costos_rutas(a)), 2)

    def f2(self, a):
        return int(len(np.unique(a)))
    
    def cons1(self, a):
        return max(self.costos_rutas(a)) - self.max_cost
    
    def cons2(self, a):
        vals, counts = np.unique(a, return_counts=True)
        return 2*len(vals) - self.num_demand
        
    def _evaluate(self, z, out, *args, **kwargs):
        
        assignment, _ = self.assign(copy(z), assign_type = self.assign_type)
        
        # print(x)
        f1 = self.f1(assignment)
        f2 = self.f2(assignment)
        g1 = self.cons1(assignment)
        g2 = self.cons2(assignment)

        out["F"] = [f1,f2]
        out["G"] = [g1, g2]
        
# Asigna, permite repetidos
class Repair_normal(Repair):
    
    def _do(self, problem, Z, **kwargs):

        Z_repaired = []
        
        for z in Z:
            eval_z = copy(z)

            _, zr = problem.assign(eval_z*1, assign_type = problem.assign_type)        
            Z_repaired.append(copy(zr.astype(bool)))

        return Z_repaired

# Asigna, no permite repetidos, si se repite, muta bits hasta que no se repita
class Repair_mutation(Repair):
    
    def __init__(self, prob= 0.5):
    
        self.prob = prob
        
        super().__init__()
    
    def _do(self, problem, Z, **kwargs):

        Z_repaired = []
        
        for z in Z:
            eval_z = copy(z)
            new_z = False
            
            while not new_z:
                _, zr = problem.assign(eval_z*1, assign_type = problem.assign_type)
                
                hash_z = hashlib.sha256(zr).hexdigest()
                
                if hash_z not in problem.explored:
                    new_z = True
                    problem.explored.add(hash_z)
                    Z_repaired.append(copy(zr.astype(bool)))
                else:
                    
                    ind_to_mutate = np.nonzero(eval_z)[0]
                    for i in ind_to_mutate:
                        if np.random.rand() < self.prob:
                            eval_z[i] = not eval_z[i]
                            
                            # encuentra otro índice para mutar
                            replace = np.random.randint(0, len(eval_z))
                            while replace in ind_to_mutate:
                                replace = np.random.randint(0, len(eval_z))

                            eval_z[replace] = not eval_z[replace]
                            

        return Z_repaired


def parse_input(entrada):
    
    # for i in range(len(entrada)):
    #     # sys.stdout.write('['+str(i)+']' + entrada[i] + ' ')
    #     sys.stdout.write(entrada[i]+ ' ')
    
    # Crear un objeto ArgumentParser
    parser = argparse.ArgumentParser()

    # Agregar argumentos esperados
    parser.add_argument('-i', '--instance', type=str, help='Ruta del archivo de la instancia')
    parser.add_argument('--seed', type=int, help='Semilla para la generación de números aleatorios')
    parser.add_argument('--cr', type=str, help='Valor para cr')
    parser.add_argument('--mu', type=str, help='Valor para mu')
    parser.add_argument('--a', type=str, help='Valor para a')
    parser.add_argument('--rep', type=str, help='Valor para rep')
    parser.add_argument('--pop', type=int, help='Valor para pop')
    parser.add_argument('--dec', type=int, help='Valor para dec')
    parser.add_argument('--prob_cr', type=float, help='Valor para prob_cr')
    parser.add_argument('--prob_mu', type=float, help='Valor para prob_mu')
    parser.add_argument('--prob_ini', type=float, help='Valor para prob_ini')


    # Dividir la cadena en palabras
    args_str = entrada.split()

    # Analizar los argumentos, deteniéndose cuando encuentra una cadena que no comienza con '-'
    args = parser.parse_args(args_str)
    
    return args



def optimizar(t_max, entrada, problem, algorithm, termination, seed):
    
    try:
        # Si la función tarda demasiado tiempo, se activará la alarma
        signal.alarm(t_max*60)  
        
        res = minimize(problem,
               algorithm = algorithm,
               copy_algorithm = True,
               seed = seed,
               save_history = False,
               termination = termination,
               verbose = False
               )

        # Si la función termina antes del límite de tiempo, cancela la alarma
        signal.alarm(0)
        
        return res
    
    # Si la alarma se activa, se captura la entrada y se guarda en un archivo
    except Exception as e:
        
        with open('time_unfeasible.txt', 'a') as file:
            file.write(entrada + '\n')

        return -1

# Manejador de la señal de alarma
def manejador_alarma(signum, frame):
    raise TimeoutError("El algoritmo tardó demasiado tiempo en ejecutarse")



def exec_algorithm(entrada, returnData = False, timeout = 6):
    # -----------------------------------------------------------------
    # PARÁMETROS DE ENTRADA
    # -----------------------------------------------------------------

    args = parse_input(entrada = entrada)
    
    # Lee los parámetros de entrada
    instance = args.instance
    seed = args.seed
    cr = args.cr
    mu = args.mu
    a = args.a
    rep = args.rep
    pop = args.pop
    dec = args.dec
    prob_cr = args.prob_cr
    prob_mu = args.prob_mu
    prob_ini = args.prob_ini
    
    
    cruza = {'un': UniformCrossover(prob=prob_cr), 
             'hun': HalfUniformCrossover(prob=prob_cr)}.get(cr)
    
    mutacion = {'bf': BitflipMutation(prob=prob_mu),
                'bfo': BitflipMutation_on(prob=prob_mu)}.get(mu)
    
    inicializacion = BinaryRandomSampling_n(prob = prob_ini)
    
    reparacion = {'rn': Repair_normal(),
                  'rm': Repair_mutation(prob = 0.5)}.get(rep)
    
    asignacion = {'cl': 'classic',
                  'ur': 'urgencies',
                  'urc': 'urgencies_cap'}.get(a)
    
    poblacion = pop
    descendencia = int(poblacion * (dec/100))
    
    
    
    # -----------------------------------------------------------------
    # LECTURA DE INSTANCIA
    # -----------------------------------------------------------------
    
    with open(instance, 'rb') as f:
        instance = pickle.load(f) 
        f.close()
    
    capacidades = instance['capacidades']
    demanda = instance['demanda']
    matriz_OD = instance['matriz_OD']
    
    
    
    
    # -----------------------------------------------------------------
    # ALGORITMO
    # -----------------------------------------------------------------
    
    algorithm = NSGA2(
    pop_size = poblacion,
    n_offsprings = descendencia,
    sampling = inicializacion,
    crossover = cruza,
    mutation = mutacion,
    eliminate_duplicates = True,
    repair = reparacion,
    )

    
    
    # -----------------------------------------------------------------
    # PROBLEMA
    # -----------------------------------------------------------------
    
    problem = MinPMedian(matriz_OD, capacidades, demanda, 
                     assign_type= asignacion,
                     max_cost = 9000)

    termination = DefaultMultiObjectiveTermination(
        xtol=0.0001,
        cvtol=1e-9,
        ftol=0.0001,
        period=100,
        n_skip=100,
        n_max_gen=500,
        n_max_evals=100000
    )
    
    # -----------------------------------------------------------------
    # EJECUCIÓN
    # -----------------------------------------------------------------
    signal.signal(signal.SIGALRM, manejador_alarma)

    t_max = timeout # Tiempo máximo de ejecución en minutos
    
    res = optimizar(t_max, 
                    entrada, 
                    problem, 
                    algorithm, 
                    termination, 
                    seed)
    
    # Si la ejecución se completó, se calcula el hipervolumen
    if res != -1 and type(res.F) != type(None):
        
        # Calcula los maximos valores de cada función objetivo
        max_num_bases = np.ceil(problem.num_demand/2)
        max_f1, max_f2 = [problem.num_demand*problem.max_cost, max_num_bases + 1]

        # Normaliza las funciones objetivo
        pf = copy(res.F)
        pf[:,0] = pf[:,0]/max_f1
        pf[:,1] = pf[:,1]/max_f2
        pf = pf[pf[:,1].argsort()]
        
        # Calcula el hipervolumen del frente de pareto
        hipervolumen = HV(ref_point=[1,1])
        #print("HV", ind(pf))
        hipervolumen_inv = 1 - hipervolumen(pf)

        if returnData:
            return hipervolumen_inv, res
        else:
            sys.stdout.write(f' Best {hipervolumen_inv} ')
    
    # Si la ejecución no se completó, devuelve un 1
    else:
        if returnData:
            return 1, res
        sys.stdout.write(f' Best 1')


if __name__ == "__main__":

    # -----------------------------------------------------------------
    # PARÁMETROS DE ENTRADA
    # -----------------------------------------------------------------
    
    entrada = ' '.join(sys.argv[1:])
    
    exec_algorithm(entrada)
    



    
