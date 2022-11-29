import pandas as pd
from itertools import permutations
import pickle
from time import time

def experimento(solver, E, J, t):
    # Carga de datos

    df = pd.read_csv('../Data/equipos_data.csv', index_col = 0)
    equipos = list(df['equipo'].unique())[:E]
    partidos = list(permutations(equipos, 2))

    filename = '../Data/list_calendar.pkl'
    lista_dias = open(filename, 'rb')
    with open(filename, 'rb') as f:
        lista_dias=pickle.load(f)

    lista_dias = lista_dias[:J]

    filename = '../Data/dic_dias_jugables_r.pkl'
    dic_dias_reales = open(filename, 'rb')
    with open(filename, 'rb') as f:
        dic_dias_reales=pickle.load(f)
        
    dic_dias_reales = {k : v for k,v in dic_dias_reales.items() if v in lista_dias}

    filename = '../Data/list_day.pkl'
    es_finde = open(filename, 'rb')
    with open(filename, 'rb') as f:
        es_finde=pickle.load(f)
    # Es finde? 1 si, 0 no

    es_finde = es_finde[:J]
    
    filename = '../Data/predicciones.pickle'
    with open(filename, 'rb') as f:
        predicciones=pickle.load(f)

    predicciones = {k : v for k,v in predicciones.items() if k[0] in partidos and k[1] in lista_dias}

    # Variables

    x = {}
    for i in partidos:
        for j in lista_dias:
            x[i, j] = solver.BoolVar(f'Partido {i} jugado el dia {j}')

    # Variables ficticias

    dl = {}
    for i in partidos: 
        dl[i] = solver.IntVar(0, len(lista_dias)-1, f'Día lectivo en el que se juega {i}')
        solver.Add(dl[i] == solver.Sum([dia * x[i,j] for dia, j in enumerate(lista_dias)]))

    dr = {}
    for i in partidos: 
        dr[i] = solver.IntVar(0, 295, f'Día real en el que se juega {i}')
        solver.Add(dr[i] == solver.Sum([dia * x[i,j] for dia, j in dic_dias_reales.items()]))

    # Restricciones

    for j in lista_dias:
        solver.Add(solver.Sum([x[i, j] for i in partidos])<=4)

    for i in partidos:
        solver.Add(solver.Sum([x[i, j] for j in lista_dias])==1)

    solver.Add(solver.Sum([x[i, lista_dias[j]] for j in range(len(lista_dias)) if es_finde[j] for i in partidos])>=0.6*len(partidos))

    for i in partidos:
        for indice, j in enumerate(lista_dias[:len(lista_dias)- len(lista_dias)//2 + 1]):
            solver.Add(solver.Sum([x[i, j1]+x[(i[1],i[0]), j1] for j1 in lista_dias[indice:indice+ len(lista_dias)//2 +1]])<=1)

    for equipo1 in equipos:
        for dia in range(1, len(lista_dias)+1):
            restriccion = solver.Constraint(-2, 2, '')
            for j in lista_dias[:dia]:
                for equipo2 in equipos:
                    if equipo1 != equipo2:
                        restriccion.SetCoefficient(x[(equipo1, equipo2), j], 1)
                        restriccion.SetCoefficient(x[(equipo2, equipo1), j], -1)

    for e in equipos:
        for i in partidos:
            for i_aux in partidos:
                if e in i and e in i_aux and i!=i_aux and not (i[0] == i_aux[1] and i[1] == i_aux[0]):
                    Y_pos = solver.BoolVar(f'Partido {i} vs partido {i_aux} Y positiva')
                    Y_neg = solver.BoolVar(f'Partido {i} vs partido {i_aux} Y negativa')
            
                    solver.Add(Y_pos + Y_neg == 1)
                    solver.Add(-1000000*Y_pos + dr[i]-dr[i_aux]>= -1000000 +3)
                    solver.Add(1000000*Y_neg + dr[i]-dr[i_aux]<= 1000000 -3)

    m = 14

    for equipo in equipos:
        for j,fecha in enumerate(lista_dias):
            if j < len(lista_dias) - m - 1:
                restriccion = solver.Constraint(1,solver.Infinity(), f'Máximo numero de días sin jugar {equipo}, fecha {j}')
                for aux in range(1,m+1):
                    k=j+aux-1

                    for i in partidos:
                        if equipo in i:
                            restriccion.SetCoefficient(x[i, lista_dias[k]], 1)

    # Función Objetivo

    objective_terms = []
    for i in partidos:
        for j in lista_dias:
            objective_terms.append(predicciones[i,j] * x[i, j])
    solver.Maximize(solver.Sum(objective_terms))

    # Resultado

    t0 = time()

    solver.SetTimeLimit(t * 1000) # Maximo de tiempo en ms
    status = solver.Solve()

    t1 = time() - t0

    return status, t1

