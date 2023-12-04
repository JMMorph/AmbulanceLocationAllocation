# Bibliotecas de uso general en el cuaderno
from myutils import *
import cugraph
import pickle
import time
import os

# Datos generales (PLTBs y hospitales)

# Directorio de datos generales
path = "../GeoData/"
escenario = '8'
out = "../GeoData/Instances/Instance_" + escenario + ".pkl"





def get_instance(path, escenario, out ='', save = False):

    inicio = time.time()
    
    # --------------------------------------------------------------- Time counter
    t_info = time.time()
    # Carga el grafo de la red de transporte de la Ciudad de México
    g = ox.load_graphml(filepath = path + "graph_transport.graphml")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(g, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)

    # Carga las pltbs
    pltbs = gpd.read_file(path + 'PLTBs.gpkg', layer = 'PLTBs_nodes')

    # --------------------------------------------------------------- Time counter
    t_info = time.time() - t_info
    
    # Serializa los datos de tipo lista
    pltbs['streets'] = pltbs['streets'].apply(json.loads)
    pltbs['oneway'] = pltbs['oneway'].apply(json.loads)
    pltbs['length'] = pltbs['length'].apply(json.loads)
    pltbs['capacity'] = pltbs['capacity'].apply(json.loads)
    pltbs['grouped'] = pltbs['grouped'].apply(json.loads)
    pltbs_grupos= [item for sublist in [ i for i in pltbs['grouped']] for item in sublist]
    pltbs_nodos = list(pltbs['node'])

    
    
    

    # Identifica todos los nodos que no se pueden alcanzar desde una PLTB
    voronoi_pltbs = nx.voronoi_cells(g, pltbs_nodos, weight='weight')
    unreachable = voronoi_pltbs["unreachable"]
    del(voronoi_pltbs)

    # Elimina los nodos que no se pueden alcanzar desde una PLTB
    g_u = g.copy()
    g_u.remove_nodes_from(unreachable)

    # --------------------------------------------------------------- Time counter
    t_temp = time.time()
    # Carga el archivo con la ubicación de los hospitales
    hospitales = gpd.read_file(path + 'Points.gpkg', layer = 'hospitales')
    # --------------------------------------------------------------- Time counter
    t_info += time.time() - t_temp

    # Identifica hospitales de interés
    hospitales_viables = hospitales.query("camas_en_a > 10")

    # Obtiene las coordenadas
    hospitales_coords = hospitales_viables.get_coordinates()

    # Obtiene las aristas más cercanas en el grafo de la red de transporte "g"
    aristas_hospitales = list(set(ox.nearest_edges(g_u, hospitales_coords.x, hospitales_coords.y)))

    # añade las aristas en doble sentido, son las que se van a descartar como PLTBs
    aristas_hospitales_d = copy(aristas_hospitales)
    aristas_hospitales_d += [(v,u,k) for u,v,k in aristas_hospitales_d if (v,u,k) in g_u.edges.keys()]








    # En esta sección se obtienen datos que varían dependiendo del incidente, como la posición de los puntos de demanda o los datos de tráfico

    # --------------------------------------------------------------- Time counter
    t_temp = time.time()
    # Carga el archivo con la ubicación de los puntos de atención
    colapsos = gpd.read_file(path + 'Points_instances.gpkg', layer = 'Instance_' + escenario)
    # --------------------------------------------------------------- Time counter
    t_info += time.time() - t_temp
    
    # Obtiene las coordenadas
    colapsos_coords = colapsos.get_coordinates()

    # Obtiene las aristas más cercanas en el grafo de la red de transporte "g", 
    # en este caso es importante mantener el orden, por eso se usa una lista en lugar de un set
    aristas_colapsos = ox.nearest_edges(g_u, colapsos_coords.x, colapsos_coords.y)

    # Añade las aristas en doble sentido, son las que se van a bloquear
    aristas_colapsos_b = list(set(copy(aristas_colapsos)))
    aristas_colapsos_b += [(v,u,k) for u,v,k in aristas_colapsos_b if (v,u,k) in g_u.edges.keys()]

    # Las aristas bloqueadsa y sus vecinas también serán descartadas como PLTBs
    aristas_colapsos_d = copy(aristas_colapsos_b)
    aristas_colapsos_d += [(u,v,0) for a in aristas_colapsos_d for u,v in g.edges(a[0])]

    # Elimina el grafo g_u
    del(g_u)
    
    
    
    
    
    
    
    # Se lee el archivo .tar con la información de tráfico para cada arista del grafo
    traffic_files = [os.path.join(path + 'Traffic/', file) for file in os.listdir(path + 'Traffic/') if file.endswith("_traffic_data.tar")]
    traffic_files.sort()
    # print(traffic_files[int(escenario)])
    
    # --------------------------------------------------------------- Time counter
    t_temp = time.time()
    traffic_data = pd.read_pickle(traffic_files[int(escenario)])
    # --------------------------------------------------------------- Time counter
    t_info += time.time() - t_temp

    # Añade la información de tráfico al grafo
    # Crear una columna con el tráfico en cada arista
    gdf_edges = gdf_edges.assign(traffic= traffic_data)


    # Crea una columna para indicar la penalización por bloqueo
    weight_max = 1e9
    block = lambda x: 1 if (x[0],x[1],0) in aristas_colapsos_b else 0
    gdf_edges['blocked'] = gdf_edges.index.to_series().apply(block)

    bby_colapsos = gdf_edges["blocked"].sum()
    # print(f'Se bloquearon {bby_colapsos} aristas por colapsos')

    gdf_edges['blocked'] += ((gdf_edges['traffic'] == 5) | (gdf_edges['traffic'] == 6))*1

    bby_traffic = gdf_edges["blocked"].sum() - bby_colapsos
    # print(f'Se bloquearon {bby_traffic} aristas por tráfico')









    # Pondera las aristas según el tráfico, momentaneamente se asigna un valor de 0 a las aristas bloqueadas
    map_vel = {1:1, 2: 2, 3: 4, 4: 6.7, 5: 0, 6: 0}
    gdf_edges['weight'] = gdf_edges.apply(lambda x: x['length'] * map_vel[ x['traffic']], axis=1)

    # Asigna el paso de todas las aristas bloqueadas a 1e9
    gdf_edges['weight'] = gdf_edges['weight'] * (1-gdf_edges['blocked']) + weight_max * gdf_edges['blocked']






    # Descarte de PLTBs
    # Aristas a descartar como PLTBs por no tener tráfico libre
    aristas_trafico_d = [(u,v,k) for u,v,k in gdf_edges[gdf_edges['traffic'] > 2].index.to_series()]

    # Aristas que se deben descartar como PLTBs
    aristas_a_descartar = list(set(aristas_hospitales_d + aristas_colapsos_d))
    pltbs_a_descartar = set()

    for u,v,k in aristas_a_descartar:
        
        # Si alguno de los nodos está en la lista de PLTBs agrupadas, los nodos head del grupo(s) se agregan a la lista de PLTBs bloqueadas 
        if u in pltbs_grupos or v in pltbs_grupos:
            block = list(pltbs[pltbs.apply( lambda x: u in x['grouped'] or v in x['grouped'], axis = 1)].get('node'))
            pltbs_a_descartar = pltbs_a_descartar.union(set(block))
    
        # Descarta pltbs que no tienen sucesores (normalmene en los extremos del bbox)
        if len(list(g.successors(u))) == 0:
            pltbs_a_descartar.add(u)
        if len(list(g.successors(v))) == 0:
            pltbs_a_descartar.add(v)
            
    # En el caso del tráfico, se descartan las aristas que no tienen tráfico libre 
    for u,v,k in aristas_trafico_d:
        
        # Si el nodo v es una PLTB, se agrega a la lista de PLTBs a descartar
        if v in pltbs_nodos:
            pltbs_a_descartar.add(v)
        if u in pltbs_nodos:
            pltbs_a_descartar.add(u)


    # print('pltbs que quedan descartadas:')
    # print('(', end='')
    # for b in pltbs_a_descartar:
    #     print(b,end=', ')
    # print(')')
    # print('Total de PLTBs descartadas:', len(pltbs_a_descartar))
    

    # Vuelve a forma el grafo de la red de transporte
    g = ox.graph_from_gdfs(gdf_nodes, gdf_edges)

    # Si se desea guardar el grafo bloqueado
    # save_graph_geopackage(g, filepath='../GeoData/graph_transport_bloqueado.gpkg', layer='grafo_bloqueado', encoding='utf-8')








    ## Cálculo de matríz origen destino

    # Grafo de transporte invertido
    gi = g.reverse()
    g_i_nx = nx.DiGraph(gi)

    edges_i = nx.to_pandas_edgelist(g_i_nx)
    edges_i = edges_i[['source','target','weight']]

    g_i_cuda = cugraph.Graph(directed=True)
    g_i_cuda.from_pandas_edgelist(edges_i, source='source', destination='target', edge_attr='weight')

    # print('Total de aristas omitidas: ',len(gdf_edges)-len(g_i_cuda.edges()))

    # Define los origenes y destinos en términos de los ids de los nodos del grafo de transporte

    # Identifica los orígenes y destinos vecinos, considerando las vías bloqueadas
    origenes = [ p for p in pltbs_nodos if p not in pltbs_a_descartar]
    destinos = colapsos['num']

    # Identifica para cada punto de colapso "c" su arista con puntos vecinos "u,v"
    # con los que se puede llegar a "c" desde "u" o "v"
    colapsos_vecinos = dict([(c,[u,v]) for c, (u,v,k) in zip(destinos, aristas_colapsos )])

    # Identifica todos los puntos de destino sin repetir
    destinos_vecinos = set([i for u,v,k in aristas_colapsos for i in [u,v]])


    # Crea un diccionario con los indices de los nodos de origen y destino
    origen_i_to_node = dict([(i,o) for i,o in enumerate(origenes)])
    origen_node_to_i = dict([(o,i) for i,o in enumerate(origenes)])

    # Mapeos de indices "num" a nodos y viceversa
    destino_num_to_node = colapsos_vecinos
    destino_node_to_num = dict([(i,c) for c, (u,v,k) in zip(destinos, aristas_colapsos ) for i in [u,v]])

    # Mapeos de indices "num" a índices reales y viceversa
    destino_i_to_num = dict([(i, d) for i,d in enumerate(destinos)])
    destino_num_to_i = dict([(d, i) for i,d in enumerate(destinos)])

    pltbs.index = pltbs['node']
    colapsos.index = colapsos['num']
    # Toma la capacidad total/2 de cada PLTB
    capacidades = [pltbs.loc[i]['t_capacity']/2 for i in origenes]


    
    
    
    # --------------------------------------------------------------- Time counter
    t_rutas_cuda = time.time()
    # Obtiene las rutas desde cada punto de origen hasta cada destino
    # Se almacena en un diccionario, en donde la llave son los destinos y el valor es un dataframe con el resultado del algoritmo de Dijkstra, de dondes e puede obtener la ruta y el costo de llegada desde todos los puntos del grafo de transporte

    rutas = dict()

    # Calcula las rutas más cortas para cada punto de atención usando el grafo invertido
    # y dijkstra, las almacena en un diccionario cuya llave es el punto de destino

    for i,d in enumerate(destinos_vecinos):
        df_rutas = cugraph.sssp(g_i_cuda, d, edge_attr='weight')
        df_rutas.set_index('vertex', inplace=True)
        rutas[d] = df_rutas.round(4).to_pandas()

    # print('costo calculado para cada punto de atención')
    
    # --------------------------------------------------------------- Time counter
    t_rutas_cuda = time.time() - t_rutas_cuda
    


    # ### Obtención de la matriz Origen Destino
    # 
    # En esta matriz, las filas son las PLTBS y las columnas los colapsos
    # 
    # filas: PLTBS
    # columnas: Colapsos

    # Crea la matriz OD
    matriz_OD = np.zeros((len(origenes), len(destinos)))

    # Para cada foco de atención "d":
    for i, d in enumerate(destinos):
        
        # Identifica cuáles son sus vecinos "u,v" por los que se puede llegar a él
        vecino_u = colapsos_vecinos[d][0]
        vecino_v = colapsos_vecinos[d][1]
        
        # De las rutas calculadas, obtiene la distancia de cada origen a "u" y a "v"
        distancias_u = rutas[vecino_u].loc[origenes]['distance'].rename('distancia_u')
        distancias_v = rutas[vecino_v].loc[origenes]['distance'].rename('distancia_v')
        
        distancias_u = distancias_u*(distancias_u < weight_max) + 10e20*(distancias_u > 10e20)
        distancias_v = distancias_v*(distancias_v < weight_max) + 10e20*(distancias_v > 10e20)
        
        # Encuentra la distancia máxima para llegar a "u" o a "v desde cada origen
        distancias_d = pd.concat([distancias_u, distancias_v], axis=1).max(axis=1)
        
        # Agrerga la columna a la matriz OD
        matriz_OD[:,i] = distancias_d.to_numpy()



    # ## Determina la demanda de cada colapso
    # 
    # Para determinar la demanda, toma el porcentaje de víctimas en cada uno de los colapsos, y toma ese porcentaje del total de ambulancias disponibles. Además garantiza que cada punto tenga al menos una demanda de 1

    victimas = np.array([colapsos.loc[destino_i_to_num[i]]['victimas'] for i in range(len(destinos))])

    # Numero total de ambulancias disponibles
    num_ambulancias = 250 
    demanda = (victimas/np.sum(victimas)) * (num_ambulancias - len(victimas))
    demanda = np.round(demanda).astype(int) + 1
    demanda = demanda.tolist()

    # if sum(demanda) > num_ambulancias:
    #     print('Demanda proporcional ajustada: ',sum(demanda))
        

    # ## Almacenamiento de la instancia
    # Posteriormente se almacenan las variables que representan a esta instancia

    instancia = {'origenes': origenes,
                'destinos': destinos,
                
                'origen_i_to_node': origen_i_to_node,
                'origen_node_to_i': origen_node_to_i,
                
                'destino_i_to_num': destino_i_to_num,
                'destino_num_to_i': destino_num_to_i,
                
                'destino_num_to_node': destino_num_to_node,
                'destino_node_to_num': destino_node_to_num,
                
                'capacidades': capacidades,
                'demanda': demanda,
                
                'matriz_OD': matriz_OD,
                'rutas': rutas
                }
    
    
    # --------------------------------------------------------------- Time counter
    t_total = time.time() - inicio
    
    
    
    
    
    # --------------------------------------------------------------- Time counter
    t_rutas_nx = time.time()
    # Calcula las rutas más cortas para cada punto de atención usando el grafo invertido
    # y dijkstra, las almacena en un diccionario cuya llave es el punto de destino
    
    rutas_nx = dict()
    for i,d in enumerate(destinos_vecinos):
        rutas_nx[d] = nx.single_source_dijkstra(g_i_nx, d, weight='weight')[0]
    
    # --------------------------------------------------------------- Time counter
    t_rutas_nx = time.time() - t_rutas_nx




    if save:
        with open(out, 'wb') as f:  
            pickle.dump(instancia, f)
            f.close()
    
    info = {'t_total': t_total,
            't_rutas_cuda': t_rutas_cuda,
            't_rutas_nx': t_rutas_nx,
            't_info': t_info,
            'bby_colapsos': bby_colapsos,
            'bby_traffic': bby_traffic}
    
    
    return instancia, info

    # with open(out, 'rb') as f:
    #     instance = pickle.load(f) 
    #     f.close()


