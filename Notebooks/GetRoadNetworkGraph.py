from myutils import *


# Carga el polígono de la ciudad de méxico 
cdmx_layer = gpd.read_file(r'../GeoData/Polygons.gpkg', layer='CDMX', encoding='utf-8')

# Si se desea aplicar un buffer, se indica el valor en metros
buffer = 0.01

filepath = "../GeoData/"
filename = "graph_transport"

if buffer:
    # Obtiene la geometría de la capa
    cdmx_polygon = cdmx_layer.geometry[0]
    # saca el buffer con shapely
    cdmx_polygon = sp.buffer(cdmx_polygon, buffer)
    # devuelve el polígono a la capa
    cdmx_layer['geometry'] = [cdmx_polygon]
    print('buffer aplicado')

# Reproyecta a WGS84 para OSMNX
cdmx_layer.to_crs(4326, inplace=True)
cdmx_polygon = cdmx_layer.geometry[0]


# Define el tipo de vialidad que se va a descargar
filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'

# Descarga el grafo de la red de transporte de la Ciudad de México 
gt = ox.graph_from_polygon(polygon = cdmx_polygon, network_type='drive', truncate_by_edge=False, custom_filter=filter,simplify=False)
gt = ox.simplification.simplify_graph(gt, track_merged=True)
gt = ox.project_graph(gt, to_crs=6369)

# Elimina los atributos que no son necesarios
gdf_nodes, gdf_edges = ox.graph_to_gdfs(gt)

gdf_edges.drop(columns=['reversed', 'junction', 'width', 'bridge', 'access', 'tunnel', 'service'], inplace=True)

new_gt = ox.graph_from_gdfs(gdf_nodes, gdf_edges)

ox.save_graph_geopackage(new_gt, filepath = filepath + filename + ".gpkg", directed=True)
ox.save_graphml(new_gt, filepath = filepath + filename + ".graphml")

# gt = ox.load_graphml(filepath="../GeoData/graph_transport.graphml")