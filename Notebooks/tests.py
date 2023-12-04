import pandas

# import subprocess
# import os

# # Define la ruta al entorno virtual de Anaconda
# ruta_entorno_anaconda = '/home/cic/anaconda3/envs/rapids-pymoo'  # Reemplaza con la ruta correcta

# # Define el comando que deseas ejecutar
# comando = [f'{ruta_entorno_anaconda}/bin/python3', 'optimize.py', 'hola', '8']

# # Define el entorno que deseas utilizar
# entorno = os.environ.copy()
# entorno['PATH'] = f'{ruta_entorno_anaconda}/bin:' + entorno['PATH']

# # Inicia el proceso con el entorno específico
# proceso = subprocess.Popen(comando, 
#                            stdout=subprocess.PIPE, 
#                            stderr=subprocess.PIPE, 
#                            env=entorno)

# # Captura la salida estándar y de errores
# stdout, stderr = proceso.communicate()

# # Imprime la salida estándar y de errores
# print("Salida estándar:")
# print(stdout.decode('utf-8'))

# print("\nErrores estándar:")
# print(stderr.decode('utf-8'))
# # import pickle
# import matplotlib.pyplot as plt

# import numpy as np

# # Para trabajar con las teselas guardadas previamiente en otra sesión
# path = '../GeoData/Instances/previo/Usable/'
# name = 'traffic_flow_16_2023-10-12_20-10-09'
# tiles = pickle.load(open(path + name + '.pkl', 'rb'))


# tile = tiles[14712][29172].astype('uint8')
# tile[...,-1] = 0

# tile[:,:,[0,1,2,3]] = tile[:,:,[0,1,3,2]].astype('uint8')

# # plt.imshow(tile)

# tt = tile.view('uint32').squeeze(-1)
# plt.imshow(tt)
# plt.show()





# mask = tile[:,:,3] > 250
# tile_s = tile[mask]

# r, g, b = [], [], []

# for pixel in tile_s:
#     r.append(pixel[0])
#     g.append(pixel[1])
#     b.append(pixel[2])
    


# # ax = plt.figure().add_subplot(projection='3d')
# # ax.scatter(r,g,b, c= [(a/255,b/255,c/255) for a,b,c in zip(r,g,b)], s=1)


# plt.scatter(r,g, c= [(a/255,b/255,c/255) for a,b,c in zip(r,g,b)], s=1)

# plt.show()