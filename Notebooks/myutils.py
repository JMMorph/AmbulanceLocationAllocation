# Bibliotecas con herramientas de geoprocesamiento

from osmnx import utils_graph
from osmnx import settings
from osmnx import utils
import geopandas as gpd
import networkx as nx
import osmnx as ox
import operator
import cugraph
import pickle
import os



# Bibliotecas para trabajar con imágenes

from rasterio.merge import merge
import matplotlib.pyplot as plt
from PIL import Image
import rasterio

# Bibliotecas para trabajar con geometrías
import shapely as sp
from shapely.geometry import Point
from shapely.geometry import LineString


# Bibliotecas de uso general

from decimal import Decimal as D
from copy import deepcopy as copy
from pathlib import Path
from io import BytesIO
import pandas as pd
import numpy as np
import pickle
import math
import json
import os


# Bibliotecas específicas para la función de descarga de tiles

import datetime

import requests
import asyncio
import aiohttp
import time

# Bibliotecas para las funciones modificadas:
from networkx.utils import groups
from heapq import heappop, heappush
from itertools import count

# Bibliotecas de optimización
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling



# -------------------------------------------------------------------------------------------- #
# ---------------------------------- FUNCIONES MODIFICADAS ----------------------------------- #
# -------------------------------------------------------------------------------------------- #
# ----------- Estas funciones fueron obtenidas de las bibliotecas y modificadas -------------- #
# ---------- para ajustarse a las necesidades del proyecto, en un futuro de hará ------------- #
# -------- una contribución a las bibliotecas para que estas funciones estén disponibles  ---- #
# -------------------------------- para todos los usuarios ----------------------------------- #
# -------------------------------------------------------------------------------------------- #




# La modificación realizada fue para añadir capas a un gpkg existente y no reescribirlo
def save_graph_geopackage(G, filepath=None, layer = "", encoding="utf-8", directed=True):
    """
    Save graph nodes and edges to disk as layers in a GeoPackage file.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    filepath : string or pathlib.Path
        path to the GeoPackage file including extension. if None, use default
        data folder + graph.gpkg
    encoding : string
        the character encoding for the saved file
    directed : bool
        if False, save one edge for each undirected edge in the graph but
        retain original oneway and to/from information as edge attributes; if
        True, save one edge for each directed edge in the graph

    Returns
    -------
    None
    
    This function is a modified version of osmnx.utils_graph.save_graph_geopackage
    """
    # default filepath if none was provided
    if filepath is None:
        filepath = Path(settings.data_folder) / "graph.gpkg"
    else:
        filepath = Path(filepath)

    # if save folder does not already exist, create it
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # if file already exists and no layer name is provided, will be overwritten
    if filepath.is_file() and not layer:
        Path(filepath).unlink()

    # convert graph to gdfs and stringify non-numeric columns
    if directed:
        gdf_nodes, gdf_edges = utils_graph.graph_to_gdfs(G)
    else:
        gdf_nodes, gdf_edges = utils_graph.graph_to_gdfs(utils_graph.get_undirected(G))
    gdf_nodes = _stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = _stringify_nonnumeric_cols(gdf_edges)
    
    if layer:
        layer += "_"

    # save the nodes and edges as GeoPackage layers
    gdf_nodes.to_file(filepath, layer = layer + "nodes", driver="GPKG", index=True, encoding=encoding)
    gdf_edges.to_file(filepath, layer = layer + "edges", driver="GPKG", index=True, encoding=encoding)
    utils.log(f"Saved graph as GeoPackage at {filepath!r}")


    
# Esta función no se modificó, pero se incluye para que se pueda usar desde el archivo principal
def _stringify_nonnumeric_cols(gdf):
    """
    Make every non-numeric GeoDataFrame column (besides geometry) a string.

    This allows proper serializing via Fiona of GeoDataFrames with mixed types
    such as strings and ints in the same column.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        gdf to stringify non-numeric columns of

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        gdf with non-numeric columns stringified
        
    This function is a modified version of osmnx.utils_graph._stringify_nonnumeric_cols
    """
    # stringify every non-numeric column other than geometry column
    for col in (c for c in gdf.columns if not c == "geometry"):
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            gdf[col] = gdf[col].fillna("").astype(str)

    return gdf

# -------------------------------------------------------------------------------------------- #
# --- Función tomada de nx.voronoi_cells y modificada para que devuelva más información ------ #
# -------------------------------------------------------------------------------------------- #

def voronoi_cells(G, center_nodes, weight="weight", path=False, pred=False, length=False):
    """Returns the Voronoi cells centered at `center_nodes` with respect
    to the shortest-path distance metric.

    If $C$ is a set of nodes in the graph and $c$ is an element of $C$,
    the *Voronoi cell* centered at a node $c$ is the set of all nodes
    $v$ that are closer to $c$ than to any other center node in $C$ with
    respect to the shortest-path distance metric. [1]_

    For directed graphs, this will compute the "outward" Voronoi cells,
    as defined in [1]_, in which distance is measured from the center
    nodes to the target node. For the "inward" Voronoi cells, use the
    :meth:`DiGraph.reverse` method to reverse the orientation of the
    edges before invoking this function on the directed graph.

    Parameters
    ----------
    G : NetworkX graph

    center_nodes : set
        A nonempty set of nodes in the graph `G` that represent the
        center of the Voronoi cells.

    weight : string or function
        The edge attribute (or an arbitrary function) representing the
        weight of an edge. This keyword argument is as described in the
        documentation for :func:`~networkx.multi_source_dijkstra_path`,
        for example.

    Returns
    -------
    dictionary
        A mapping from center node to set of all nodes in the graph
        closer to that center node than to any other center node. The
        keys of the dictionary are the element of `center_nodes`, and
        the values of the dictionary form a partition of the nodes of
        `G`.

    Examples
    --------
    To get only the partition of the graph induced by the Voronoi cells,
    take the collection of all values in the returned dictionary::

        >>> G = nx.path_graph(6)
        >>> center_nodes = {0, 3}
        >>> cells = nx.voronoi_cells(G, center_nodes)
        >>> partition = set(map(frozenset, cells.values()))
        >>> sorted(map(sorted, partition))
        [[0, 1], [2, 3, 4, 5]]

    Raises
    ------
    ValueError
        If `center_nodes` is empty.

    References
    ----------
    .. [1] Erwig, Martin. (2000),"The graph Voronoi diagram with applications."
        *Networks*, 36: 156--163.
        https://doi.org/10.1002/1097-0037(200010)36:3<156::AID-NET2>3.0.CO;2-L

    """
    # Determine the shortest paths from any one of the center nodes to
    # every node in the graph.
    #
    # This raises `ValueError` if `center_nodes` is an empty set.
    preds = {node: '' for node in G.nodes()}
    # paths = nx.multi_source_dijkstra_path(G, center_nodes, weight=weight,pred=pred)
    lengths, paths = multi_source_dijkstra(G, center_nodes, weight=weight,pred=preds)
    
    # Determine the center node from which the shortest path originates.
    nearest = {v: p[0] for v, p in paths.items()}
    # Get the mapping from center node to all nodes closer to it than to
    # any other center node.
    cells = groups(nearest)
    # We collect all unreachable nodes under a special key, if there are any.
    unreachable = set(G) - set(nearest)
    if unreachable:
        cells["unreachable"] = unreachable
    
    if length:
        cells["length"] = lengths
    if path:
        cells["path"] = paths
    if pred:
        cells["pred"] = preds

    
    return cells


# -------------------------------------------------------------------------------------------- #
# -- Función tomada de nx.multi_source_dijkstra y modificada para que devuelva más información #
# -------------------------------------------------------------------------------------------- #

def multi_source_dijkstra(G, sources, target=None, cutoff=None, weight="weight", pred=[]):
    """Find shortest weighted paths and lengths from a given set of
    source nodes.

    Uses Dijkstra's algorithm to compute the shortest paths and lengths
    between one of the source nodes and the given `target`, or all other
    reachable nodes if not specified, for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    target : node label, optional
        Ending node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list
        If target is None, returns a tuple of two dictionaries keyed by node.
        The first dictionary stores distance from one of the source nodes.
        The second stores the path from one of the sources to that node.
        If target is not None, returns a tuple of (distance, path) where
        distance is the distance from source to target and path is a list
        representing the path from source to target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.multi_source_dijkstra(G, {0, 4})
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 1
    4: 0
    >>> path[1]
    [0, 1]
    >>> path[3]
    [4, 3]

    >>> length, path = nx.multi_source_dijkstra(G, {0, 4}, 1)
    >>> length
    1
    >>> path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    https://code.activestate.com/recipes/119466/

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    multi_source_dijkstra_path
    multi_source_dijkstra_path_length

    """
    if not sources:
        raise ValueError("sources must not be empty")
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    if target in sources:
        return (0, [target])
    weight = _weight_function(G, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _dijkstra_multisource(
        G, sources, weight, paths=paths, cutoff=cutoff, target=target, pred=pred
        
    )
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError as err:
        raise nx.NetworkXNoPath(f"No path to {target}.") from err


# -------------------------------------------------------------------------------------------- #
# -- Función tomada de nx._dijkstra_multisource y modificada para que devuelva más información #
# -------------------------------------------------------------------------------------------- #

def _dijkstra_multisource(G, sources, weight, pred=None, paths=None, cutoff=None, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.

    weight: function
        Function with (u, v, data) input that returns that edge's weight
        or None to indicate a hidden edge

    pred: dict of lists, optional(default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored.

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node.
        If None, paths are not stored.

    target : node label, optional
        Ending node for path. Search is halted when target is found.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    Returns
    -------
    distance : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.

    Raises
    ------
    NodeNotFound
        If any of `sources` is not in `G`.

    Notes
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

    """
    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        seen[source] = 0
        push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            cost = weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
                elif pred is not None and vu_dist == u_dist:
                    pred[u].append(v)
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return dist

def _weight_function(G, weight):
    """Returns a function that returns the weight of an edge.

    The returned function is specifically suitable for input to
    functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.

    Parameters
    ----------
    G : NetworkX graph.

    weight : string or function
        If it is callable, `weight` itself is returned. If it is a string,
        it is assumed to be the name of the edge attribute that represents
        the weight of an edge. In that case, a function is returned that
        gets the edge weight according to the specified edge attribute.

    Returns
    -------
    function
        This function returns a callable that accepts exactly three inputs:
        a node, an node adjacent to the first one, and the edge attribute
        dictionary for the eedge joining those nodes. That function returns
        a number representing the weight of an edge.

    If `G` is a multigraph, and `weight` is not callable, the
    minimum edge weight over all parallel edges is returned. If any edge
    does not have an attribute with key `weight`, it is assumed to
    have weight one.

    """
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)







# -------------------------------------------------------------------------------------------- #
# ----------------------------------- FUNCIONES PROPIAS -------------------------------------- #
# -------------------------------------------------------------------------------------------- #
# ---------------------- Estas funciones fueron creadas para el proyecto --------------------- #
# -------------------------------------------------------------------------------------------- #





# Función para pasar a minúsculas odo el contenido de un GeoDataFrame
def lower_content(df,filepath = "", layer = "dataframe", encoding="utf-8"):
    
    """
    Lowercase all the content of a GeoDataFrame

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        gdf to lowercase
    filepath : string or pathlib.Path
        path to the GeoPackage file including extension. if None, do not save
    layer : string
        layer name to save the GeoDataFrame as in the GeoPackage file, if saving
    encoding : string
        the character encoding for the saved file
    

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        gdf with lowercased content
    """
    
    df.rename(columns = dict([(c,c.lower()) for c in df.columns]), inplace = True)
    
    # lower case every non-numeric column other than geometry column
    for col in (c for c in df.columns if not c == "geometry"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].str.lower()
            
    if filepath and layer:
        df.to_file(filepath, layer = layer, driver="GPKG", encoding=encoding)

    return df





# Función para añadir las coordenadas de los centroides de las geometrías
# a un GeoDataFrame, en crs EPSG:4326
def add_latlon(gdf, filepath = "", layer = "dataframe", encoding="utf-8"):
    """
    Add latitude and longitude columns to a GeoDataFrame from its geometry column.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        gdf to add latitudes and longitudes to
    filepath : string or pathlib.Path
        path to the GeoPackage file including extension. if None, do not save
    layer : string
        layer name to save the GeoDataFrame as in the GeoPackage file, if saving
    encoding : string
        the character encoding for the saved file

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        gdf with latitudes and longitudes added
    """
    
    crs = gdf.crs
    gdf.to_crs(epsg=4326, inplace=True)
    gdf["lat"] = gdf["geometry"].apply(lambda point: point.centroid.y)
    gdf["lon"] = gdf["geometry"].apply(lambda point: point.centroid.x)
    gdf.to_crs(crs, inplace=True)
    
    if filepath and layer:
        gdf.to_file(filepath, layer = layer, driver="GPKG", encoding=encoding)
    
    return gdf





# Función para convertir columnas a tipo numérico
def column_to_numeric(df, columns, filepath = "", layer = "dataframe", encoding="utf-8"):
    
    """
    Function to convert columns to numeric type.
    
    Parameters:
    -----------
    df: geopandas.GeoDataFrame
        GeoDataFrame to convert.
    columns: dict
        Dictionary with column names as keys and numeric type as values.
    filepath: str
        Path to save the GeoDataFrame.
    layer: str
        Name of the layer to save.
    encoding: str
        Encoding of the GeoDataFrame.
        
    Returns:
    --------
    df: geopandas.GeoDataFrame
        GeoDataFrame with the columns converted.
    

    """
    
    force_numeric = lambda x: x if x.replace('.','').isnumeric() else 0
    
    for column in columns:
        
        
        df[column].fillna(0, inplace=True)
        
        try:
        
            if df[column].dtype == 'object' or df[column].dtype == 'str':
                df[column] = df[column].apply(force_numeric)
            
            df[column] = df[column].astype(columns[column])
            
        except:
            df[column] = df[column].astype(columns[column])
            
    if filepath and layer:
        df.to_file(filepath, layer = layer, driver="GPKG", encoding=encoding)

    return df





# Función para convertir coordenadas geodésicas a coordenadas de tiles
def lat_lon_to_tile_zxy(lat, lon, zoom_level):
    
    """
    Convert latitude and longitude to tile coordinates
    
    Parameters
    ----------
    lat : float
        Latitude value
    lon : float
        Longitude value
    zoom_level : int
        Zoom level value
    
    Returns
    -------
    list
        List of tile coordinates [z, x, y]
        
    Raises
    ------
    ValueError
        If zoom level value is out of range [0, 22]
    ValueError
        If latitude value is out of range [-85.051128779807, 85.051128779806]
    ValueError
        If longitude value is out of range [-180.0, 180.0]
        
    Addapted from the code provided by TomTom in https://developer.tomtom.com/maps-api/maps-api-documentation/zoom-levels-and-tile-grid
    
    """
    
    # Define ranges
    min_zoom_level = 0
    max_zoom_level = 22
    min_lat = -85.051128779807
    max_lat = 85.051128779806
    min_lon = -180.0
    max_lon = 180.0

    # Check input values of zoom level, latitude and longitude
    if (zoom_level is None or not isinstance(zoom_level, (int, float))
        or zoom_level < min_zoom_level
        or zoom_level > max_zoom_level ):
        
        raise ValueError(
            f"Zoom level value is out of range [{min_zoom_level}, {max_zoom_level}]"
        )

    if lat is None or not isinstance(lat, (int, float)) or lat < min_lat or lat > max_lat:
        raise ValueError(f"Latitude value is out of range [{min_lat}, {max_lat}]")

    if (lon is None or not isinstance(lon, (int, float))
        or lon < min_lon
        or lon > max_lon ):
        
        raise ValueError(f"Longitude value is out of range [{min_lon}, {max_lon}]")

    z = int(zoom_level)
    xy_tiles_count = 2**z
    x = int(((lon + 180.0) / 360.0) * xy_tiles_count)
    y = int(((1.0 - math.log( math.tan((lat * math.pi) / 180.0) + 1.0 / math.cos((lat * math.pi) / 180.0)) /math.pi)/ 2.0)* xy_tiles_count)

    return [z,x,y]





# Función para convertir coordenadas de tiles a coordenadas geodésicas
def tile_zxy_to_lat_lon(zoom_level, x, y):
    '''
    Convert tile coordinates to latitude and longitude
    
    Parameters
    ----------
    zoom_level : int
        Zoom level value
    x : int
        Tile x value
    y : int
        Tile y value
        
    Returns
    -------
    list
        List of latitude and longitude [lat, lon]
        
    Raises
    ------
    ValueError
        If zoom level value is out of range [0, 22]
    ValueError
        If tile x value is out of range [0, 2^zoom_level - 1]
    ValueError
        If tile y value is out of range [0, 2^zoom_level - 1]
    
    Addapted from the code provided by TomTom in https://developer.tomtom.com/maps-api/maps-api-documentation/zoom-levels-and-tile-grid
    
    '''
    
    
    min_zoom_level = 0
    max_zoom_level = 22

    if (zoom_level is None or not isinstance(zoom_level, (int, float))
        or zoom_level < min_zoom_level
        or zoom_level > max_zoom_level ):
        
        raise ValueError(
            f"Zoom level value is out of range [{min_zoom_level}, {max_zoom_level}]"
        )

    z = int(zoom_level)
    min_xy = 0
    max_xy = 2**z - 1

    if x is None or not isinstance(x, (int, float)) or x < min_xy or x > max_xy:
        raise ValueError(
            f"Tile x value is out of range [{min_xy}, {max_xy}]"
        )

    if y is None or not isinstance(y, (int, float)) or y < min_xy or y > max_xy:
        raise ValueError(
            f"Tile y value is out of range [{min_xy}, {max_xy}]"
        )

    lon = (x / 2**z) * (360.0) - (180.0)

    n = (math.pi) - (2.0 * math.pi * y) / (2**z)
    lat = (180.0 / math.pi) * (math.atan(0.5 * (math.exp(n) - math.exp(-n))))

    return [lat, lon]
    # return [float(lat), float(lon)]





def tile_zxy_to_lat_lon_bbox(zoom_level, x, y):
    '''
    Find the bounding box of a tile in latitude and longitude
    
    Parameters
    ----------
    zoom_level : int
        Zoom level value
    x : int
        Tile x value
    y : int
        Tile y value
        
    Returns
    -------
    list
        List of latitude and longitude [lat1, lon1, lat2, lon2]
        
    Raises
    ------
    ValueError
        If zoom level value is out of range [0, 22]
    ValueError
        If tile x value is out of range [0, 2^zoom_level - 1]
    ValueError
        If tile y value is out of range [0, 2^zoom_level - 1]
        
    Addapted from the code provided by TomTom in https://developer.tomtom.com/maps-api/maps-api-documentation/zoom-levels-and-tile-grid
    
    '''
    
    
    min_zoom_level = 0
    max_zoom_level = 22

    if (zoom_level is None or not isinstance(zoom_level, (int, float))
        or zoom_level < min_zoom_level
        or zoom_level > max_zoom_level):
        
        raise ValueError(
            f"Zoom level value is out of range [{min_zoom_level}, {max_zoom_level}]"
        )

    z = int(zoom_level)
    min_xy = 0
    max_xy = 2**z - 1

    if x is None or not isinstance(x, (int, float)) or x < min_xy or x > max_xy:
        raise ValueError(f"Tile x value is out of range [{min_xy}, {max_xy}]")

    if y is None or not isinstance(y, (int, float)) or y < min_xy or y > max_xy:
        raise ValueError(f"Tile y value is out of range [{min_xy}, {max_xy}]")

    lon1 = (x / 2**z) * 360.0 - 180.0

    n1 = math.pi - (2.0 * math.pi * y) / 2**z
    lat1 = (180.0 / math.pi) * math.atan(0.5 * (math.exp(n1) - math.exp(-n1)))


    lon2 = ((x + 1) / 2**z) * 360.0 - 180.0

    n2 = math.pi - (2.0 * math.pi * (y + 1)) / 2**z
    lat2 = (180.0 / math.pi) * math.atan(0.5 * (math.exp(n2) - math.exp(-n2)))

    # return f"{lat1}/{lon1}/{lat2}/{lon2}"
    return [lat1, lon1, lat2, lon2]





# Función para convertir un arreglo de canales RGBA a un arreglo monobanda
def rgba_to_mono(tile):
    
    '''
    Function to convert a RGBA tile to a monochromatic tile
    
    Parameters
    ----------
    tile : numpy.ndarray
        Tile to convert
        
    Returns
    -------
    numpy.ndarray, dtype = np.uint8
        Monochromatic tile
        
    '''
    
    # Máscara para identificar los pixeles con información
    mask = tile[:,:,3] > 250

    # Obtener los canales RGB y aplicar la máscara
    tile2 = tile[:,:,0:3] * np.expand_dims(mask, axis = 2)
    
    # Posteriza la imágen a 4 valores por canal
    tile2 = tile2/(255/4)
    tile2 = tile2.round() * (255/4)
    
    # Aplica la fórmula 5*R + G + 10*B
    res = 5*tile2[...,1] + tile2[...,0] + 10*tile2[...,2]

    # Define los colores de la imagen
    # colores = [0, 206, 1594, 1913, 1721, 1211, 4335]
    colores = [0, 1657, 1848,1530, 212, 1148,  4080  ]
    
    # Define las tolerancias para cada color
    tols = np.array([0, 10, 10, 1, 50, 10, 10])

    # Calcula la diferencia entre la imágen y los colores
    dif  = np.abs(res - np.expand_dims(colores,axis=[1,2]))
    
    # Obtiene el índice del color más cercano
    ind_min = np.argmin(dif,axis=0).astype(np.intp)
    
    # Asigna el color más cercano sólo si la diferencia es menor a la tolerancia
    ima =  ind_min * (dif.min(axis=0) <= np.expand_dims(tols,axis=[1,2]).take(ind_min))
    
    return ima.astype(np.uint8)





# FUnción para guardar por filas las teselas en un archivo GeoTIFF 
# a partir de un diccionario de teselas
def save_geotif(tiles,tiles_data, path, name):
    '''
    Function to save a dictionary of tiles as a GeoTIFF file
    
    Parameters
    ----------
    tiles : dict
        Dictionary of tiles
    tiles_data : dict
        Dictionary with information about the tiles
    path : str
        Path to save the GeoTIFF file
    name : str
        Name of the GeoTIFF file
        
    Returns
    -------
    None
    
    Produces
    -------
    GeoTIFF file
        
    '''
    
    zoom = tiles_data['zoom']
    imsize = tiles_data['imsize']
    tiles_x = tiles_data['tiles_x']

    lower = tiles_data['lower']
    upper = tiles_data['upper']
    inf_izq = tiles_data['inf_izq']
    sup_der = tiles_data['sup_der']
    sup_izq = tiles_data['sup_izq']
    inf_der = tiles_data['inf_der']
    rgba = tiles_data['rgba']
    
    # Si no se especifica un nombre, se utiliza el mismo nombre que el archivo de datos
    if not name:
        name = tiles_data['name']
        
    
    
    ### ----- Creación de filas como raster ----- ###
    
    # Ruta completa de la carpeta "temp" para almacenar los rasters individuales
    # de cada fila temporalmente
    
    ruta_temp = os.path.join(path, "temp")

    # Verificar si la carpeta "temp" existe
    if not os.path.exists(ruta_temp):
        # Si no existe, crearla
        os.makedirs(ruta_temp)
    
    # Lista para almacenar los nombres de los rasters individuales
    filas_temp = []

    # Recorre los índices de las filas
    for j in range(upper[2], lower[2]+1):
        
        # Lista para almacenar las teselas de la fila
        fila = []
        
        # Recorre los índices de las columnas
        for i in range(lower[1], upper[1]+1):
            
            # Si la tesela existe, se agrega a la fila
            if tiles[i][j]:
                # tiles[i][j][0][0] = np.array([0,0,0,255])
                # tiles[i][j][-1][-1] = np.array([255,0,0,255])
                fila.append(tiles[i][j]['array'].astype(np.uint8))
            
            # Si no existe, se agrega una tesela vacía
            else:
                if rgba:
                    fila.append(np.zeros((imsize,imsize,4), dtype=np.uint8))
                
                else:
                    fila.append(np.zeros((imsize,imsize), dtype=np.uint8))
                
        
        fila = np.hstack(fila)
        
        # Calcula las coordenadas en lat, lon de las esquinas de la tesela    
        y1, x1, y2, x2 = tile_zxy_to_lat_lon_bbox(zoom,lower[1],j)
        x1, x2 = sup_izq[1], sup_der[1]
        
        # def from_bounds(west, south, east, north, width, height):
        transformacion = rasterio.transform.from_bounds(x1,y2,x2,y1,imsize*tiles_x,imsize)
        
        if rgba:
            banda = 4
        else:
            banda = 1
        
        row_name = f'{path}temp/row_{j}.tif'
        new_dataset = rasterio.open(row_name, 'w', driver='GTiff',
                                    height = fila.shape[0], width = fila.shape[1],
                                    count=banda, dtype=str(fila.dtype),
                                    crs='EPSG:4326',
                                    transform=transformacion)

        if rgba:
            new_dataset.write(np.rollaxis(fila, axis=2) )
        else:
            # new_dataset.write(np.rollaxis(np.expand_dims(fila,axis=-1), axis=2))
            new_dataset.write(fila,1)
            
        new_dataset.close()
        del(new_dataset)
        
        filas_temp.append(row_name)
    
    
    
    
    ### ----- Combinación de filas en una sola imagen ----- ###

    # Cargar los rasters individuales
    rasters = []

    for ruta_raster in filas_temp:
        rasters.append(rasterio.open(ruta_raster) )

    # Combinar los rasters en una sola imagen
    merged, out_transform = merge(rasters)

    # Actualizar la información de georreferenciación si es necesario
    out_profile = rasters[0].profile
    out_profile.update(transform=out_transform, width=merged.shape[2], height=merged.shape[1])

    # Guardar la imagen combinada
    with rasterio.open(path+name+'.tif', 'w', **out_profile) as dst:
        dst.write(merged)
        written = True

    if written:
        print(f'Imagen guardada como "{path + name}.tif"')
        # Eliminar los rasters
        del(merged)
        del(rasters)
        for ruta_raster in filas_temp:
            os.remove(ruta_raster)
    
    return 





# Función para generar la url de descarga de las teselas
def get_url(x,y,zoom, imsize, apikey):
    
    url = f'https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{zoom}/{x}/{y}.png?tileSize={imsize}&key=' + apikey
    return url





# Función para hacer la consulta y descargar las teselas de tráfico
async def get(query, session, tiles, zoom, imsize, apiKey, attempts, failed):
    try:
        
        x = query['x']
        y = query['y']
        
        if query['query'] == 1:
            url_query = get_url(query['x'], query['y'], zoom, imsize, apiKey)
            
            async with session.get(url=url_query) as response:
                
                if response.status != 200:
                    for _ in range(attempts):
                        await asyncio.sleep(1)  # Esperar 1 segundo antes de volver a intentar
                        
                        # Hacer un nuevo intento con la misma URL
                        async with session.get(url=url_query) as new_response:
                            if new_response.status == 200:
                                resp = await new_response.read()
                                image = np.array(Image.open(BytesIO(resp)))
                                tiles[x][y] = image
                                #print(f'Imagen obtenida: {x},{y}')
                                break  # Si se obtiene una respuesta exitosa, salir del bucle
                            
                    else:
                        print(f'Error al obtener imagen para {x},{y}. Se agotaron los intentos.')
                        failed.append(query)
                else:
                    resp = await response.read()
                    image = np.array(Image.open(BytesIO(resp)))
                    tiles[x][y] = image
                    #print(f'Imagen obtenida: {x},{y}')
        else:
            tiles[x][y] = 0
            
            
    except Exception as e:
        print("Unable to get url {} due to {}.".format(query, e.__class__))
        
        
  
        
        
# Función asíncrona para mandar las consultas de descarga de teselas
async def queries_async(urls, tiles, zoom, imsize, apiKey, attempts, failed):
    async with aiohttp.ClientSession() as session:
        ret = await asyncio.gather(*[get(url, session, tiles, zoom, imsize, apiKey, attempts, failed) for url in urls])
    print(f'Descarga de teselas inalizada, teselas faltantes: {len(failed)}')





# Función para descargar las teselas de tráfico a partir de un polígono
async def get_traffic_tiles(polygon, zoom = 16, imsize = 512, apiKey = '', rgba= False, 
                            save = True, saveGeometry = False, path = '', name = ''):
    
    '''
    Function to download traffic flow tiles from TomTom API
    
    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to download the tiles from
    zoom : int
        Zoom level value
    imsize : int
        Size of the tiles
    apiKey : str
        TomTom API key
    rgba : bool
        If True, the tiles are saved as RGBA images
    save : bool
        If True, the tiles are saved as a dictionary
    saveGeometry : bool
        If True, the tiles are saved as a GeoPackage file
    path : str
        Path to save the files
    name : str  
        Name of the files
        
    Returns
    -------
    tiles : dict
        Dictionary of tiles, where the keys are the x and y coordinates of the tiles
        and the values are the tiles as numpy arrays, and the coordinates of the upper
        left and lower right corners of the tiles as latitude and longitude, or an empty
        string if the tile does not intersect with the polygon
        
    tiles_data : dict
        Dictionary with information about the tiles, including:
        zoom : int
            Zoom level value
        imsize : int
            Size of the tiles
        tiles_x : int
            Number of tiles in the x axis
        tiles_y : int
            Number of tiles in the y axis
        total_tiles : int
            Total number of tiles
        tiles_used : int
            Number of tiles that intersect with the polygon
        lower : list
            Lower left corner of the bounding box of the polygon as tile coordinates
        upper : list
            Upper right corner of the bounding box of the polygon as tile coordinates
        inf_izq : tuple
            Lower left corner of the bounding box of the polygon as latitude and longitude
        sup_der : tuple
            Upper right corner of the bounding box of the polygon as latitude and longitude
        sup_izq : tuple
            Upper left corner of the bounding box of the polygon as latitude and longitude
        inf_der : tuple
            Lower right corner of the bounding box of the polygon as latitude and longitude
        name : str
            Name of the file
        fecha : str
            Date and time of the download
        failed : list
            List of tiles that could not be downloaded
    
    '''
    
    # Obtiene el bounding box del polígono
    (lon1 , lat1, lon2, lat2) = polygon.bounds
    
    
    ### ----- Generación de las teselas ----- ###

    # Primera y última tesela
    lower = lat_lon_to_tile_zxy(lat1, lon1, zoom)
    upper = lat_lon_to_tile_zxy(lat2, lon2, zoom)


    # Número de teselas por lado
    tiles_x = (upper[1] - lower[1] + 1)
    tiles_y = (lower[2] - upper[2] + 1)
    total_tiles = tiles_x*tiles_y
    # print(f'Teselas totales: {tiles_x}x{tiles_y} = {tiles_x*tiles_y}')

    # Coordenadas en lat, lon de las esquinas de la imagen, en el sistema de referencia WGS84
    inf_izq = tile_zxy_to_lat_lon(zoom, lower[1], lower[2] + 1)
    sup_der = tile_zxy_to_lat_lon(zoom, upper[1] + 1, upper[2])
    sup_izq = (sup_der[0], inf_izq[1])
    inf_der = (inf_izq[0], sup_der[1])

    # Cálculo de la resolución de las teselas
    circunferencia = 40075017
    resolucion_tiles = circunferencia/(2**zoom) # metros
    resolucion_pixeles = resolucion_tiles/imsize # metros/pixel

    print('Coordenadas reales:', inf_izq, sup_der)
    print('resolucion en pixeles', resolucion_pixeles, 'metros/pixel')
    
    # Genera la geometría de las teselas
    tiles_bbox = {'x':[], 'y':[], 'geometry':[], 'intersect':[]}
    

    # Identifica las teselas que intersectan con el polígono de la ciudad de méxico
    # para descargar únicamente las teselas que contienen información de la ciudad
    queries = []

    # Recorre los índices de las teselas
    for i in range(lower[1], upper[1]+1):
        for j in range(upper[2], lower[2]+1):
    
            y1,x1,y2,x2 = tile_zxy_to_lat_lon_bbox(zoom,i,j)
            
            point1 = (x1,y1)
            point2 = (x1,y2)
            point3 = (x2,y2)
            point4 = (x2,y1)
            
            # Genera la geometría de la tesela
            tile = sp.Polygon([point1,point2,point3,point4])
            
            # Almacena la geometría de la tesela
            tiles_bbox['x'].append(i)
            tiles_bbox['y'].append(j)
            tiles_bbox['geometry'].append(tile)
            
            # Si la tesela intersecta con el polígono de la ciudad de méxico
            if tile.intersects(polygon):
                
                tiles_bbox['intersect'].append(1)
                
                # Marca la tesela para descargar
                queries.append({'x': i, 'y':j, 'query': 1, 'sup_izq': (y1,x1), 'inf_der': (y2,x2)})
            
            else:
                
                tiles_bbox['intersect'].append(0)
                
                # Marca la tesela para no descargar
                queries.append({'x': i, 'y':j, 'query': 0, 'sup_izq': [], 'inf_der': []})
            
    bbox = sp.Polygon([(inf_izq[1], inf_izq[0]), (sup_der[1], inf_izq[0]), (sup_der[1], sup_der[0]), (inf_izq[1], sup_der[0])])
    tiles_bbox['x'].insert(0, 0)
    tiles_bbox['y'].insert(0, 0)
    tiles_bbox['geometry'].insert(0, bbox)
    tiles_bbox['intersect'].insert(0,-1)

    # Convierte las teselas a un GeoDataFrame
    tiles_bbox = gpd.GeoDataFrame(tiles_bbox, crs=4326)

    # Número de teselas a descargar
    tiles_used = sum([ 1 if i['query'] == 1 else 0 for i in queries])
    
    
    
    ### ----- Descarga de las teselas ----- ###

    # Lista para almacenar las imágenes
    tiles = dict([(i, dict([ (j,'') for j in range(upper[2], lower[2]+1)] )) for i in range(lower[1], upper[1]+1)])
    attempts = 5
    failed = []
    
    result = await queries_async(queries, tiles, zoom, imsize, apiKey, attempts, failed)
    
    # Obtiene la fecha y hora actual
    fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not name:
        name = f'traffic_flow_{zoom}_{fecha}'
        
    
    # Guarda el bounding box de las teselas como un GeoPackage
    if saveGeometry:
        tiles_bbox.to_file(path + 'tiles_bbox.gpkg', layer='tiles', driver="GPKG", encoding="utf-8")
    
    del(tiles_bbox)
    
    
    
    ### ----- Guardado de las teselas ----- ###
    
    # Si no se desea el formato RGBA, se convierten las imágenes a monocromáticas
    if not rgba:
        
        # Aplica la función para convertir las imágenes a monocromáticas
        for q in queries:
            if q['query'] == 1:
                tiles[q['x']][q['y']] = {'array' : rgba_to_mono(tiles[q['x']][q['y']]), 
                                        'sup_izq': q['sup_izq'], 
                                        'inf_der': q['inf_der']}
    else:
        
        # Aplica la función para guardar las imágenes como RGBA
        for q in queries:
            if q['query'] == 1:
                tiles[q['x']][q['y']] = {'array' : tiles[q['x']][q['y']], 
                                        'sup_izq': q['sup_izq'], 
                                        'inf_der': q['inf_der']}
        
    
    tiles_data = {'zoom': zoom, 
                  'imsize': imsize, 
                  'tiles_x': tiles_x,
                  'tiles_y': tiles_y,
                  'total_tiles': total_tiles, 
                  'tiles_used': tiles_used, 
                  'lower': lower,
                  'upper': upper,
                  'inf_izq': inf_izq,
                  'sup_der': sup_der,
                  'sup_izq': sup_izq,
                  'inf_der': inf_der,
                  'resolucion_pixeles': resolucion_pixeles,
                  'name': name,
                  'fecha': fecha,
                  'failed': failed,
                  'rgba': rgba}
    
    # Para guardar el diccionario con las teselas y los metadatos
    if save:
        with open(path + name + '.pkl', 'wb') as file:
            pickle.dump(tiles, file)
            print(f'Teselas guardadas como "{path + name}_tiles.pkl"')
            
        with open(path + name + '_metadata.pkl', 'wb') as file:
            pickle.dump(tiles_data, file)
            print(f'Metadatos guardados como "{path + name}_tiles_data.pkl"')
    
    return tiles, tiles_data





# Función para obtener el punto medio de una línea
def middle_point(geom):
    '''
    Function to get the middle point of a line (with an offset of 3 meters to the right)
    
    Parameters
    ----------
    geom : shapely.geometry.LineString
        Line to get the middle point
        Requires at least two points
    
    Returns
    -------
    shapely.geometry.Point
        Middle point of the line (with an offset of 3 meters to the right)
    
    '''
    
    # Obtiene las coordenadas de los puntos que forman la línea
    lats, lons = geom.xy
    lats = lats.tolist()
    lons = lons.tolist()
    
    
    
    # Si la línea tiene dos puntos, 'a' es el punto medio y 'b' es el punto final
    if len(lats) == 2:
        mid_point = [sum(lats)/2, sum(lons)/2]
        a = np.array(mid_point)
        b = np.array([lats[1], lons[1]])
    
    # Si la línea tiene más de dos puntos, 'a' es el punto medio de los dos puntos centrales
    # y b es el punto central posterior a 'a'
    else:
        
        middle = int(len(lats)/2)
        a = np.array([sum(lats[middle-1:middle+1]), sum(lons[middle-1:middle+1])])/2
        b = np.array([lats[middle], lons[middle]])
    
    # Calcula el vector que va de 'a' a 'b'
    vec_ab = b - a
    
    # Calcula el vector perpendicular a 'ab'
    perpendicular = np.array([vec_ab[1], -vec_ab[0]])
    
    # Normaliza el vector perpendicular y lo traslada al punto 'a', lo desplaza 3 metros
    perpendicular = 3*perpendicular/np.linalg.norm(perpendicular)
    perpendicular += a
    
    # Regresa el punto extremo del vector perpendicular
    return Point(perpendicular[0], perpendicular[1])





# Función para muestrar una imagen geo-referenciada a partir de un punto
def sample_raster(point, zoom, tiles, imsize, tiles2 = ''):
    
    '''
    Function to sample a raster image from a point
    
    Parameters
    ----------
    point : shapely.geometry.Point
        Point to sample the raster image
    zoom : int
        Zoom level of the raster image
    tiles : dict
        Dictionary of tiles
    imsize : int
        Size of the tiles
    tiles2 : dict, optional
        Dictionary of tiles to modify
    
    Returns
    -------
    int
        Mode of the values in the square window around the point
    
    '''
    
    
    # Obtiene las coordenadas del punto
    lat = point.y
    lon = point.x
    
    # Obtiene las coordenadas de la tesela correspondiente
    z, x,y = lat_lon_to_tile_zxy(lat, lon, zoom)
    lat1, lon1 = tiles[x][y]['sup_izq']
    lat2, lon2 = tiles[x][y]['inf_der']
    
    # Normaliza las coordenadas, considerando las esquinas van de 0 a imsize
    centro_x = math.floor(imsize * (lon - lon1)/(lon2 - lon1))
    centro_y = math.floor(imsize * (lat - lat1)/(lat2 - lat1))
    
    # Pixel correspondiente al centroide
    c = [centro_y, centro_x]
    
    # radio de la ventana
    r = 5
    window = tiles[x][y]['array'][max(c[0] - r, 0): min(imsize - 1, c[0] + r),
                                  max(c[1] - r, 0): min(imsize - 1, c[1] + r)]
    
    # Cuenta la frecuencia de cada valor en la ventana
    vals, counts = np.unique(window, return_counts=True)
    
    # Si sólo hay un valor, asigna el valor más común
    if len(vals) == 1:
        most_common = vals[0]
        
        # Si el valor más común es 0, asigna 1 (libre)
        if most_common == 0:
            most_common = 1

    # Si hay múltiples valores:
    else:
        
        # Encuentra el valor más común en la ventana, sin considerar el 0
        most_common = vals[np.argmax(counts[1::]) + 1]
    
    # Si se adjunta una segunda matriz, asigna el valor más común a la ventana correspondiente
    if tiles2:
        tiles2[x][y]['array'][max(c[0] - r, 0): min(imsize - 1, c[0] + r),
                             max(c[1] - r, 0): min(imsize - 1, c[1] + r)] = most_common
    
    return most_common





# Función para obtener los datos de tráfico de una red de calles
def get_traffic_data(g, tiles, tiles_data, save = True, path = '', name = ''):
    '''
    Function to get traffic data from a street network
    
    Parameters
    ----------
    g : networkx.MultiDiGraph
        Street network
    tiles : dict
        Dictionary of tiles
    tiles_data : dict
        Dictionary with information about the tiles
    save : bool
        If True, the traffic data is saved as a pickle file
    path : str
        Path to save the file
    name : str
        Name of the file
    
    Returns
    -------
    traffic_data : pandas.DataFrame
        DataFrame with the traffic data
    
    '''
    
    
    zoom = tiles_data['zoom']
    imsize = tiles_data['imsize']
    rgba = tiles_data['rgba']
    
    if not name:
        name = tiles_data['name']
        
    if rgba:
        tiles = copy(tiles)
        for i in tiles:
            for j in i:
                if isinstance(tiles[i][j], np.ndarray):
                    tiles[i][j]['array'] = rgba_to_mono(tiles[i][j]['array'])
  
    
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(g, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
    
    # Obtiene los puntos medios de las aristas para sacar la muestra
    gdf_edges['middle_point'] = gdf_edges.apply(lambda row : middle_point(row['geometry']), axis = 1)

    # Crea un objeto GeoSeries con los puntos medios de las aristas
    gdf_edges_mid = gdf_edges['middle_point'].copy()
    gdf_edges_mid.rename('geometry', inplace=True)

    # Por defecto estan en el sistema de referencia del grafo, por lo que se convierten a WGS84
    # para poder obtener las teselas correspondientes
    gdf_edges_mid.crs = gdf_edges.crs
    gdf_edges_mid = gdf_edges_mid.to_crs(4326)
    
    traffic_data = gdf_edges_mid.apply(lambda row : sample_raster(row, zoom, tiles, imsize))
    traffic_data = pd.DataFrame(traffic_data)
    if 'geometry' in traffic_data.columns:
        traffic_data.rename(columns={'geometry':0}, inplace=True)
    print("Datos de tráfico obtenidos")
    
    if save:
        traffic_data.to_pickle(f'{path}{name}_traffic_data.tar')
        print('Datos de tráfico guardados como: ' + f'{path}{name}_traffic_data.tar')
    
    return traffic_data
    
 
 
    
    
# Función para obtener la ruta más corta desde un dataframde de predecesores,
# normalmente obtenido después de aplicar el algoritmo de Dijkstra con CuGraph
def get_route_from_df(df, id):
    """
    Function to get the route from a DataFrame
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the shortest path data
    id : int
        ID of the destination vertex
    
    Returns
    -------
    length : float
        Length of the shortest path
    answer : list
        List of vertices in the shortest path
        
    Raises
    ------
    ValueError
        If the vertex is not in the result set
        
    Examples
    --------
    
    a = get_route_from_df(rutas[d_test[0]],o_test)

    print('(',end='')
    for i in a[1]:
        print(i,end=', ')

    print(')')
    
    """
    
    # Obtiene el DataFrame correspondiente al vértice
    ddf = df.loc[id]
    
    # Si el DataFrame está vacío, el vértice no está en el conjunto de resultados
    if len(ddf) == 0:
        raise ValueError("The vertex ", id, " is not in the result set")
    
    # Obtiene el predecesor del vértice
    pred = int(df.loc[id]["predecessor"])

    # Inicializa la lista de vértices y la longitud de la ruta
    answer = [id]
    length = float(df.loc[id]["distance"])

    # Mientras el predecesor no sea -1, agrega el predecesor a la lista de vértices
    while pred != -1:
        answer.append(pred)
        pred = int(df.loc[pred]["predecessor"])

    # Invierte la lista de vértices
    answer.reverse()

    # Regresa la longitud y la lista de vértices
    return length, answer






 
# Función para obtener la ruta más corta desde un diccionario de predecesores,
# normalmente obtenido después de aplicar el algoritmo de Dijkstra multi-origen
# con networkx
def get_route_from_dicts(preds, dists, origenes, destino):
    """
    Function to get the route from a dictionary
    
    Parameters
    ----------
    preds : dict
        Dictionary with the predecessors in the shortest path data
    dists : dict
        Dictionary with the shortest path data
    origenes : list
        List of source vertices
    destino : int
        ID of the destination vertex
        
    Returns
    -------
    length : float
        Length of the shortest path
    answer : list
        List of vertices in the shortest path
        
    Raises
    ------
    ValueError
        If the vertex is not in the result set
        
    
    """
    
    # Obtiene el predesesor inmediato del destino
    ddf = preds.get(destino, 0)
    
    # Si no hay predecesor, el vértice no está en el conjunto de resultados
    if len(ddf) == 0:
        raise ValueError("The vertex ", destino, " is not in the result set")
    
    # Obtiene el predecesor del destino
    pred = preds[destino][0]

    # Inicializa la lista de vértices y la longitud de la ruta
    answer = [destino]
    length = dists[destino]

    # Mientras el predecesor no sea un origen, agrega el predecesor a la lista de vértices
    while pred not in origenes:
        answer.append(pred)
        pred = preds[pred][0]

    # Agrega el origen a la lista de vértices
    answer.append(pred)
    
    # Invierte la lista de vértices
    answer.reverse()

    # Regresa la longitud y la lista de vértices de la ruta
    return length, answer   





class BinaryRandomSampling_p(Sampling):

    def __init__(self, prob=0.5):
    
        self.prob = prob
        
        super().__init__()
        
    
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < self.prob).astype(bool)
    
    
    
    
    
class BinaryRandomSampling_n(Sampling):

    def __init__(self, prob=0.5):
    
        self.prob = prob
        
        super().__init__()
        
    
    def _do(self, problem, n_samples, **kwargs):
        
        val = np.zeros((n_samples, problem.n_var))
        for i in range(val.shape[0]):
            
            index = np.random.choice(range(problem.n_var), max(int(problem.n_var*self.prob), 3), replace=False)
            val[i][index] = 1
        
        return val.astype(bool)
   
   
    


class BitflipMutation_on(Mutation):

    def _do(self, problem, X, **kwargs):
        prob_var = self.get_prob_var(problem, size=1)
        
        Xp = np.copy(X)
        
        Xp_mutated = []
        for x in Xp:
            
            # Encuentra los índices a mutar
            ind_to_mutate = np.nonzero(x)[0]
            for i in ind_to_mutate:
                if np.random.rand() < prob_var:
                    
                    # invierte el bit

                    x[i] = not x[i]
                    
                    # encuentra otro índice para mutar
                    replace = np.random.randint(0, problem.n_var)
                    
                    # SE asegura que no esté en la lista de índices a mutar
                    while replace in ind_to_mutate:
                        replace = np.random.randint(0, problem.n_var)

                    x[replace] = not x[replace]
            
            # Agrega el individuo mutado a la lista
            Xp_mutated.append(x)

        return np.array(Xp_mutated)












#  Funcion para identificar un elemento dominado
def select_dominated(a,b):
    """
    Function to select the dominated element
    
    Parameters
    ----------
    a : tuple
        Tuple with the ID and the objective values of the first element
    b : tuple
        Tuple with the ID and the objective values of the second element
        
    Returns
    -------
    tuple
        Tuple with the ID and the objective values of the dominated element
    
    Based on: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    
    """
    ge = all(map(operator.ge, a[1], b[1]))
    le = all(map(operator.le, a[1], b[1]))
    # return dominated
    return b if le else a if ge else 'indifferent'

# Función para obtener las soluciones no dominadas
def paretoFront(a):
    """
    Function to get the non-dominated solutions
    
    Parameters
    ----------
    a : list
        List of tuples with the ID and the objective values of the elements
    
    based on: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    
    """
    b = copy(a)
    if len(a) > 1:
        for i in range(len(a)):
            for j in range(i,len(a)):
                if i != j:
                    try:
                        b.remove(select_dominated(a[i],a[j]))
                    except:
                        ""
    return b


# Función para obtener el frente de Pareto fusionado
def merge_pareto_fronts(resultados):
    """
    Function to merge the Pareto fronts
    
    Parameters
    ----------
    resultados : list
        List of the results of the optimization
        
    Returns
    -------
    merged_objectives : numpy.ndarray
        Array with the merged objectives
    merged_designs : numpy.ndarray
        Array with the merged designs
    
    """
    
    objective_space = []
    design_space = []
    assigned = []
    
    for i in resultados:
        res = i[1]
        objective_space += res.F.tolist()
        design_space += res.X.tolist()
        
        problem = i[2]
        assigned += [problem.assign(sol)[0] for sol in res.X]
        
    
    objective_space = [(i, r) for i, r in enumerate(objective_space)]
    design_space = [(i, r) for i, r in enumerate(design_space)]
    assigned_space = [(i, r) for i, r in enumerate(assigned)]
    
    merged_ind = paretoFront(objective_space)
    indices = [i[0] for i in merged_ind]
    
    merged_objectives = np.array([objective_space[i][1] for i in indices])
    merged_designs = np.array([design_space[i][1] for i in indices])
    merged_assigned = np.array([assigned_space[i][1] for i in indices])
    
    sorted_args = np.argsort(merged_objectives[:,1])
    
    merged_objectives = merged_objectives[sorted_args]
    merged_designs = merged_designs[sorted_args]
    merged_assigned = merged_assigned[sorted_args]
    
    return merged_objectives, merged_designs, merged_assigned
        
        
        
        
        


# Función para obtener la ruta más corta desde un diccionario de predecesores,
def get_instance(path_source = '../GeoData/', 
                 path_scenario = 'Results/scenario/',
                 file_scenario = 'demand_points.gpkg',
                 layer_scenario = 'demand_points',
                 ambulances = 250,
                 traffic_file = None,
                 traffic = 'previous',
                 traffic_history = None,
                 date = None,
                 out ='Instance', 
                 saveGraph = False,
                 saveInstance = False,
                 verbose = False):

    """
    Function to generate the instance of the problem
    
    Parameters
    ----------
    path_source : str
        Path to the source files
    path_scenario : str
        Path to the scenario files
    file_scenario : str
        Name of the scenario file
    layer_scenario : str
        Name of the layer in the scenario file
    ambulances : int
        Number of ambulances
    traffic_file : str
        Path to the traffic file if known
    traffic : str
        If 'current', the traffic is obtained from the TomTom API
        If 'previous', the traffic history is used
    traffic_history : str
        Path to the traffic history files
    date : datetime.datetime
        Date and time of the scenario
    out : str
        Name of the output files
    saveGraph : bool
        If True, the graph is saved as a graphml file
    
    """

    # Carga el grafo de la red de transporte de la Ciudad de México
    g = ox.load_graphml(filepath = path_source + "graph_transport.graphml")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(g, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)

    # Carga las pltbs
    pltbs = gpd.read_file(path_source + 'PLTBs.gpkg', layer = 'PLTBs_nodes')
    
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



    # Carga el archivo con la ubicación de los hospitales
    hospitales = gpd.read_file(path_source + 'Points.gpkg', layer = 'hospitales')

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

    # Carga el archivo con la ubicación de los puntos de atención
    colapsos = gpd.read_file(path_scenario + file_scenario, layer = layer_scenario)
    
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
    
    if verbose:
        print('Grafo de transporte, bases candidatas y hospitales cargados')
    
    
    
    
    
    
    
    # Se obtiene la información de tráfico
    
    # Se usa el archivo
    if traffic_file:
        traffic_data = pd.read_pickle(traffic_file)
        if traffic == 'current':
            print('se está usando el archivo de tráfico: ', traffic_file)
    
    else:
        # Si se desea obtener el tráfico de un archivo histórico
        if traffic == 'previous':
            if not traffic_history:
                e = 'Error: No se especificó la ruta con el historial de tráfico'
                print(e)
                return e
            
            if not date:
                date = datetime.datetime.now()
                print('Fecha actual: ', date)
                
            horas = [7,8,9,11,13,15,17,19,20,21]

            # Obtenemos la hora mas cercana a la hora actual
            hora_cercana = min(horas, key=lambda x:abs(x-(date.hour + date.minute/60)))
            
            # Ahora que se tiene la fecha, se buscan los archivos de tráfico 
            traffic_files = [file for file in os.listdir(traffic_history) if file.endswith("_traffic_data.tar")]

            data_to_use = []
            dates_to_use = []

            # evalua todos los archivos de trafico y se queda con los que sean del mismo dia de la semana y con la hora mas cercana
            for f in traffic_files:
                # Fecha y hora del archivo
                data = f.split('_')
                date_f = data[3]
                time_f = data[4]
                year, month, day = date_f.split('-')
                hour, minute, second = time_f.split('-')
                date_f = datetime.datetime(int(year), int(month), int(day))
                hour_f = round(int(hour)+int(minute)/60, 2)
                
                # Si el dia de la semana es el mismo y la hora es cercana
                if date.weekday() == date_f.weekday() and (abs(hora_cercana - hour_f) <= 0.5 or abs(date.hour - hour_f) <= 0.5):
                    # Lee y agrega el archivo
                    traffic_data = pd.read_pickle(traffic_history+f)
                    data_to_use.append(traffic_data)
                    dates_to_use.append(date.combine(date_f, datetime.time(int(hour), int(minute), int(second))))

            if verbose:
                print('Se usan los siguientes archivos de tráfico: ')
                for d in dates_to_use:
                    print(d)
            
            # Obtiene el promedio de los datos
            traffic_data = sum(data_to_use)/len(data_to_use)
            traffic_data = round(traffic_data).astype(int)
            
        
        # Si se desea obtener el tráfico actual
        elif traffic == 'current':
            # # Carga el polígono de la ciudad de méxico, aplica un buffer de 100 mts y lo proyecta a WGS84
            cdmx = gpd.read_file(path_source+'Polygons.gpkg', layer='CDMX', encoding='utf-8')
            polygon = cdmx.geometry[0].buffer(100)
            cdmx['geometry'] = polygon
            cdmx.to_crs(4326, inplace=True)
            polygon = cdmx.geometry[0]

            # Api key de TomTom
            with open(path_source + 'tomtom_apikey.txt', 'r') as f:
                apiKey = f.read()
                f.close()
                
            zoom = 16
            imsize = 512

            # Obtiene las teselas
            tiles, tiles_data  = asyncio.run(get_traffic_tiles(polygon, zoom, imsize, apiKey, save = False, 
                                                  saveGeometry = False, path=path_scenario, rgba=False))

            name = tiles_data['name']

            # Obtiene el tráfico para cada arista según las teselas
            traffic_data = get_traffic_data(g, tiles, tiles_data, save = True, path=path_scenario, name=name)
            del(tiles)
    
    if verbose:
        print('Datos de tráfico obtenidos')

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

    # Vuelve a forma el grafo de la red de transporte
    g = ox.graph_from_gdfs(gdf_nodes, gdf_edges)

    # Si se desea guardar el grafo bloqueado
    if saveGraph:
        save_graph_geopackage(g, filepath=path_scenario + 'graph_transport_bloqueado.gpkg', layer='grafo_bloqueado', encoding='utf-8')
        
        if verbose:
            print('Grafo bloqueado guardado')
    
    if verbose:
        print('Gráfo con tráfico y aristas bloqueadas creado')









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


    
    
    
    # Obtiene las rutas desde cada punto de origen hasta cada destino
    # Se almacena en un diccionario, en donde la llave son los destinos y el valor es un dataframe con el resultado del algoritmo de Dijkstra, de dondes e puede obtener la ruta y el costo de llegada desde todos los puntos del grafo de transporte

    rutas = dict()

    # Calcula las rutas más cortas para cada punto de atención usando el grafo invertido
    # y dijkstra, las almacena en un diccionario cuya llave es el punto de destino

    for i,d in enumerate(destinos_vecinos):
        df_rutas = cugraph.sssp(g_i_cuda, d, edge_attr='weight')
        df_rutas.set_index('vertex', inplace=True)
        rutas[d] = df_rutas.round(4).to_pandas()

    if verbose:
        print('Rutas calculadas')
    # print('costo calculado para cada punto de atención')
    






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

    # Sustituye valores de 0 por 10e20
    matriz_OD += 10e20*(matriz_OD == 0)

    if verbose:
        print('Matriz OD calculada')








    # ## Determina la demanda de cada colapso
    # 
    # Para determinar la demanda, toma el porcentaje de víctimas en cada uno de los colapsos, y toma ese porcentaje del total de ambulancias disponibles. Además garantiza que cada punto tenga al menos una demanda de 1

    victimas = np.array([colapsos.loc[destino_i_to_num[i]]['victimas'] for i in range(len(destinos))])

    # Numero total de ambulancias disponibles
    num_ambulancias = ambulances
    demanda = (victimas/np.sum(victimas)) * (num_ambulancias - len(victimas))
    demanda = np.round(demanda).astype(int) + 1
    demanda = demanda.tolist()





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
    
    if verbose:
        print('Instancia creada')
        
        
    if saveInstance:
        with open(path_scenario + out+'.pkl', 'wb') as f:  
            pickle.dump(instancia, f)
            f.close()
        
        if verbose:
            print('Instancia guardada como: ', path_scenario + out+'.pkl')
    
    return instancia

    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    print('Bibliotecas y funciones cargadas correctamente')
