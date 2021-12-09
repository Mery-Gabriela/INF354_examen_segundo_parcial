from os import read
import random
import numpy as np
import csv

numero_nodos = 5
numero_poblacion = 8
max_poblacion = 30
seleccion_individuos = 4
mutacion_probabilidad = 0.2
distance_map = []

with open('agenteviajero.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        distance_map.append(
            np.array(list(map(int, row)))
        )

def individuo(min, max):
    return [random.randint(min, max) for _ in range(numero_nodos)]

def newpoblacion():
    return [np.random.permutation(numero_nodos) for _ in range(numero_poblacion)]

def funcion_objetivo(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance

def permutacion_valida(tmp):
    arr = [0, 0, 0, 0, 0]
    for x in tmp:
        arr[x] = arr[x] + 1
    tiene_repetidos = False
    for x in arr:
        if x != 1:
            tiene_repetidos = True
    return not tiene_repetidos

def seleccion_y_reproduccion(poblacion):
    evaluacion = [(funcion_objetivo(i), i) for i in poblacion]
    evaluacion = sorted(evaluacion, key=lambda pair: pair[0])
    evaluacion = [i[1] for i in evaluacion]
    poblacion = evaluacion
    selected = evaluacion[(len(evaluacion) - seleccion_individuos):]
    puntoCambio = random.randint(1, numero_nodos - 1)
    for i in range(len(evaluacion) - seleccion_individuos):
        padre = random.sample(selected, 2)
        mutado = np.array(poblacion[i], copy=True)
        mutado[:puntoCambio] = padre[0][:puntoCambio]
        mutado[puntoCambio:] = padre[1][puntoCambio:]
        if permutacion_valida(mutado):
            insertar_sin_repeticion(poblacion, mutado)
    return poblacion

def mutacion(poblacion):
    for i in range(len(poblacion) - seleccion_individuos):
        if random.random() <= mutacion_probabilidad:
            mutado = np.array(poblacion[i], copy=True)
            x1 = random.randint(0, numero_nodos - 1)
            x2 = x1
            while x2 == x1:
                x2 = random.randint(0, numero_nodos - 1)
            mutado[x1], mutado[x2] = mutado[x2], mutado[x1]
            poblacion = insertar_sin_repeticion(poblacion, mutado)
    return poblacion

def insertar_sin_repeticion(poblacion, mutado):
    for x in poblacion:
        if (x==mutado).all():
            return poblacion
    poblacion.append(mutado)
    return poblacion

def main():
    poblacion = newpoblacion()
    print(poblacion)
    for _ in range(1000):
        poblacion = seleccion_y_reproduccion(poblacion)
        poblacion = mutacion(poblacion)
        evaluacion = [(funcion_objetivo(i), i) for i in poblacion]
        evaluacion = sorted(evaluacion, key=lambda pair: pair[0])
        poblacion = [i[1] for i in evaluacion][:max_poblacion]
    print(evaluacion[0])
if __name__ == "__main__":
    main()