from functools import cached_property
import geopandas as gpd
import shapely
import numpy as np


class StreamNode:
    """
    This class represents a node in a stream network. It stores information about its coordinates,
     stream order, and connections to other nodes.

    Attributes:
        coordinates (tuple): The coordinates of the node.
        node_info (dict): A dictionary containing information about all nodes in the stream network.
        stream_order (int): The stream order of the node.
        from_tdx (bool): Indicates whether the node is from a TDX source.
        original_stream_order (int): The original stream order of the node.
        source_nodes (list): A list of source nodes connected to this node.
        source_streams (list): A list of source streams connected to this node.
        target_stream (Stream): The target stream connected to this node.
        target_coordinates (tuple): The coordinates of the target stream.

    Methods:
        __init__(
            self, coordinates, node_info, stream_order=1, from_tdx=False,
            target_coordinates=None, source_coordinates=None
        ):
            Initializes a new StreamNode instance.

            Args:
                coordinates (tuple): The coordinates of the node.
                node_info (dict): A dictionary containing information about all nodes in the stream network.
                stream_order (int, optional): The stream order of the node. Default is 1.
                from_tdx (bool, optional): Indicates whether the node is from a TDX source. Default is False.
                target_coordinates (tuple, optional): The coordinates of the target stream. Default is None.
                source_coordinates (tuple, optional): The coordinates of the source stream. Default is None.

        update_stream_order(self):
            Updates the stream order of the node based on the stream order of its source nodes.

            Returns:
                int: The updated stream order of the node.

        add_source_node(self, source_coordinates):
            Adds a source node to the source node list.

            Args:
                source_coordinates (tuple): The coordinates of the source node.

        add_target_coordinates(self, target_coordinates):
            Sets the target coordinates of the node.

            Args:
                target_coordinates (tuple): The coordinates of the target node.

        target_node(self):
            Returns the target node connected to this node.

            Returns:
                StreamNode: The target node connected to this node, or None if there is no target node.
    """

    def __init__(
            self, coordinates, node_info: dict, stream_order: int = 1,
            from_tdx: bool = False,
            target_coordinates=None, source_coordinates=None,
    ):
        if coordinates not in node_info:
            node_info[coordinates] = self
        self.node_info = node_info
        self.coordinates = coordinates
        self.from_tdx = from_tdx
        self.stream_order = stream_order
        self.original_stream_order = stream_order
        self.source_nodes = []
        self.source_streams = []
        self.target_stream = None
        self.target_coordinates = target_coordinates
        self.add_source_node(source_coordinates)

    def update_stream_order(self):
        """
        Update the stream order based on the stream orders of source nodes.

        This method iterates through the source nodes and checks their stream orders.
        The current stream order is updated if a higher stream order is found.
        If the current stream order is equal to the stream order of a source node, it is incremented.
        Finally, the updated stream order is stored in the object.

        Returns:
            int: The updated stream order.
        """
        current_stream_order = 0
        for source_node in self.source_nodes:
            if source_node.stream_order > current_stream_order:
                current_stream_order = source_node.stream_order
            elif source_node.stream_order == current_stream_order:
                current_stream_order += 1
        if current_stream_order > self.stream_order:
            self.stream_order = current_stream_order
        return self.stream_order

    def add_source_node(self, source_coordinates):
        """
        Source nodes are nodes that feed into this node. If this node has more than one source node,
         then this node is a merge point for two or more streams.

        This method adds a source node to the list of source nodes in the Graph object.

        Parameters
        ----------
        source_coordinates : object
            The coordinates of the source node to be added.

        Returns
        -------
        None
        """
        if source_coordinates is not None:
            self.source_nodes.append(self.node_info[source_coordinates])

    def add_target_coordinates(self, target_coordinates):
        """
        This method adds the coordinates of this node's target node.

        Parameters
        ----------
        target_coordinates : numpy.array
            The coordinates of the target.

        """
        if target_coordinates is not None:
            self.target_coordinates = target_coordinates

    @cached_property
    def target_node(self):
        """
        The node that this node feeds into.

        Returns:
            The target node if self.target_coordinates is not None, otherwise None.
        """
        if self.target_coordinates is None:
            return None
        else:
            return self.node_info[self.target_coordinates]


class NodeGenerator:
    """
    NodeGenerator class is responsible for generating nodes based on linestrings.

    Args:
        new_line_strings (list): List of line strings representing our model's waterways.
        old_line_strings (list): List of line strings representing tdx-hydro waterways.
        old_stream_order (int): The TDX-hydro stream order for the old_line_strings.

    Attributes:
        node_info (dict): Dictionary to store information about each node.
        source_node_coordinates (set): Set of source node coordinates for the entire network.

    Methods:
        investigate_line_string(line_string, stream_order=1, from_tdx=False):
            Investigates a line string and generates nodes based on it.

        add_node(coordinates, stream_order, target_coordinates, source_coordinates, from_tdx):
            Adds a new node to the node_info dictionary or updates an existing one.

        update_source_node_coordinates(node):
            Updates the set of source node coordinates.
    """

    def __init__(self, new_line_strings, old_line_strings, old_stream_order):
        self.node_info = {}
        self.source_node_coordinates = set()
        for line_string in old_line_strings:
            self.investigate_line_string(
                line_string=line_string.reverse(), stream_order=old_stream_order, from_tdx=True
            )
        for line_string in new_line_strings:
            self.investigate_line_string(line_string=line_string, stream_order=1, from_tdx=False)

    def investigate_line_string(self, line_string, stream_order=1, from_tdx=False):
        """
        Parameters
        ----------
        line_string : LineString
            The LineString object representing the line to investigate.

        stream_order : int, optional
            The stream order value to assign to the nodes created along the line.
            Default is 1.

        from_tdx : bool, optional
            Flag indicating whether the nodes are coming from TDX.
            Default is False.
        """
        line_string_coords = line_string.coords
        source_coords = None
        target_coords = line_string_coords[0]
        # The first node represents the source of the stream. If the stream actually has a source, it will be added
        # during that streams investigation.
        for ind, coords in enumerate(line_string_coords[:-1]):
            target_coords = line_string_coords[ind + 1]
            self.add_node(
                coordinates=coords, stream_order=stream_order, from_tdx=from_tdx,
                target_coordinates=target_coords, source_coordinates=source_coords,
            )
            source_coords = coords
        self.add_node(
            coordinates=target_coords, stream_order=stream_order, from_tdx=from_tdx,
            target_coordinates=None, source_coordinates=source_coords
        )

    def add_node(self, coordinates, stream_order, target_coordinates, source_coordinates, from_tdx):
        """
        Parameters
        ----------
        coordinates : tuple
            The coordinates of the node to be added.
        stream_order : int
            The stream order of the node.
        target_coordinates : list
            A list of coordinates representing the target nodes.
        source_coordinates : list
            A list of coordinates representing the source nodes.
        from_tdx : bool
            Flag indicating if the node is from TDX.

        """
        if coordinates not in self.node_info:
            node = StreamNode(
                coordinates=coordinates, node_info=self.node_info, stream_order=stream_order,
                from_tdx=from_tdx, target_coordinates=target_coordinates, source_coordinates=source_coordinates
            )
        else:
            node = self.node_info[coordinates]
            node.add_source_node(source_coordinates)
            node.add_target_coordinates(target_coordinates)
        self.update_source_node_coordinates(node)

    def update_source_node_coordinates(self, node: StreamNode):
        """
        This method updates the set of source_node_coordinates.

        Parameters
        ----------
        node : StreamNode
            The node whose coordinates will be updated.
        """
        # Because nodes are added iteratively, a node will start with either 0 or 1 source nodes.
        # That amount can change as the lines are iterated through, but it will only ever increase by 1 per
        # iteration, so if a node has source nodes, there will be an iteration where it has 1 source node, and at that
        # point we will remove it.
        # The coordinates that remain at the end of that process will be the true source nodes for the network,
        # and when we later generate the streams in the network, we will start at those points.
        coords = node.coordinates
        if len(node.source_nodes) == 0:
            self.source_node_coordinates.add(coords)
        elif len(node.source_nodes) == 1:
            if coords in self.source_node_coordinates:
                self.source_node_coordinates.remove(coords)


class Stream:
    """
    Class: Stream

    The Stream class represents a stream in a network, where only the first and last node in the stream
    will intersect other streams in the network. It keeps track of the nodes in the stream,
     the order of the stream, and other properties related to the stream.

    Attributes:
    - stream_id (int): The unique identifier for the stream.
    - from_tdx (float): The multipliers of the stream from its source.
    - nodes (list): The list of StreamNode objects in the stream.
    - stream_order (int): The order of the stream.
    - stream_dict (dict): A dictionary that maps stream_ids to Stream objects.
        Used for tracking all streams in the network.

    Methods:
    - __init__(self, start_node: StreamNode, stream_dict: dict):
        Initializes a new Stream object with a starting StreamNode and a stream_dict.
    - investigate_node(self):
        Investigates the stream by traversing through nodes and updating properties like source_streams and from_tdx.
    - target_stream(self):
        Cached property that returns the stream_id of the target stream.
    - source_streams(self):
        Cached property that returns the stream_ids of the source streams.
    - line_string(self):
        Cached property that returns a shapely LineString object representing
            the coordinates of the nodes in the stream.
    """
    stream_id = 0

    def __init__(self, start_node: StreamNode, stream_dict: dict):
        self.from_tdx = start_node.from_tdx
        self.nodes = [start_node]
        self.stream_order = start_node.update_stream_order()
        self.stream_dict = stream_dict
        self.stream_id = Stream.stream_id
        start_node.target_stream = self.stream_id
        self.stream_dict[self.stream_id] = self
        Stream.stream_id += 1
        self.investigate_node()

    def investigate_node(self):
        current_node = self.nodes[0].target_node
        # Iterate through the nodes until a node with multiple sources is found
        # (ie where the current stream intersects another stream).
        # Only investigate from nodes as long as they agree on from_tdx
        # (tdx_streams start from tdx nodes and end at tdx nodes, model streams start at model nodes,
        # but can end at a tdx node, but they should have at most 1 tdx node.)
        # A stream is from_tdx if all of its nodes are from_tdx.
        if current_node is not None:
            current_node.source_streams.append(self.stream_id)
            current_node.update_stream_order()
            self.from_tdx *= current_node.from_tdx
            self.nodes.append(current_node)
            while (len(current_node.source_nodes) == 1 and current_node.target_node is not None
                   and (current_node.from_tdx == self.from_tdx)):
                current_node = current_node.target_node
                current_node.update_stream_order()
                current_node.source_streams.append(self.stream_id)
                self.from_tdx *= current_node.from_tdx
                self.nodes.append(current_node)

    @cached_property
    def target_stream(self):
        # The stream this stream feeds
        return self.nodes[-1].target_stream

    @cached_property
    def source_streams(self):
        # Any streams that feed this stream
        return self.nodes[0].source_streams

    @cached_property
    def line_string(self):
        coords = [node.coordinates for node in self.nodes]
        return shapely.LineString(coords)


class StreamGenerator:
    """
    Class responsible for generating all streams in the network.

    Parameters
    ----------
    node_generator : NodeGenerator
        An instance of NodeGenerator class.
    tdx_stream_id : int, optional
        The stream id.
    old_target : int, optional
        The old target value.
    old_sources : list[int], optional
        List of old source values.

    Attributes
    ----------
    stream_dict : dict
        A dictionary containing stream information.
    seen_nodes : set
        A set containing already seen nodes.
    tdx_stream_id : int
        The stream id.
    old_target : int
        The old target value.
    old_sources : list[int]
        List of old source values.

    Methods
    -------
    make_gdf()
        Create and return a GeoDataFrame.
    investigate_node(node)
        Investigate a given node.
    """

    def __init__(
            self, node_generator: NodeGenerator, tdx_stream_id: int = 0,
            old_target: int = -1, old_sources: list[int] = ()
    ):
        Stream.stream_id = 0
        self.stream_dict = {}
        self.seen_nodes = set()
        self.tdx_stream_id = tdx_stream_id
        self.investigate_source_node_coordinates(node_generator)
        self.old_target = old_target
        self.old_sources = old_sources

    @cached_property
    def gdf(self) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(
            [
                {
                    'stream_id': stream_id,
                    'from_tdx': stream.from_tdx,
                    'source_stream_ids': np.array(stream.source_streams, dtype=np.int32),
                    'target_stream_id': stream.target_stream, 'stream_order': stream.stream_order,
                    'tdx_stream_id': self.tdx_stream_id,
                    'tdx_target_id': self.old_target,
                    'tdx_source_ids': np.array(self.old_sources, dtype=np.int32),
                    'geometry': stream.line_string
                }
                for stream_id, stream in self.stream_dict.items()
            ], crs=4326
        )
        gdf['stream_id'] = gdf['stream_id'].astype(np.int32)
        gdf['from_tdx'] = gdf['from_tdx'].astype(bool)
        gdf['stream_order'] = gdf['stream_order'].astype(np.uint8)
        gdf['tdx_stream_id'] = gdf['tdx_stream_id'].astype(np.int32)
        gdf['tdx_target_id'] = gdf['tdx_target_id'].astype(np.int32)
        gdf['target_stream_id'] = gdf['target_stream_id'].fillna(-1).astype(np.int32)
        return gdf

    def investigate_source_node_coordinates(self, node_generator: NodeGenerator):
        # Iterate through each of the true source nodes, generating their streams
        for ind, node_coordinates in enumerate(node_generator.source_node_coordinates):
            self.investigate_node(node_generator.node_info[node_coordinates])

    def investigate_node(self, node: StreamNode):
        # A node can only be investigated if all of its sources have been investigated. When a source for a node is
        # investigated, we add the generated stream to node.source_streams. So we can investigate a node when
        # len(node.source_nodes) == len(node.source_streams)

        if len(node.source_nodes) == len(node.source_streams):
            self.seen_nodes.add(node.coordinates)
            stream = Stream(node, stream_dict=self.stream_dict)
            if len(stream.nodes) > 1:
                final_node = stream.nodes[-1]
                self.investigate_node(final_node)
            else:
                stream_id = stream.stream_id
                self.stream_dict.pop(stream_id)
