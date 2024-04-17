from functools import cached_property
import geopandas as gpd
import shapely


class StreamNode:
    def __init__(
            self, coordinates, node_info: dict, stream_order: int=1, from_tdx: bool=False,
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
        if source_coordinates is not None:
            self.source_nodes.append(self.node_info[source_coordinates])

    def add_target_coordinates(self, target_coordinates):
        if target_coordinates is not None:
            self.target_coordinates = target_coordinates

    @cached_property
    def target_node(self):
        if self.target_coordinates is None:
            return None
        else:
            return self.node_info[self.target_coordinates]


class NodeGenerator:
    def __init__(self, new_line_strings, old_line_strings, old_stream_order):
        self.node_info = {}
        self.source_node_coordinates = set()
        for line_string in old_line_strings:
            self.investigate_line_string(line_string, old_stream_order, True)
        for line_string in new_line_strings:
            self.investigate_line_string(line_string)

    def investigate_line_string(self, line_string, stream_order=1, from_tdx=False):
        line_string_coords = line_string.coords
        source_coords = None
        target_coords = line_string_coords[0]
        for ind, coords in enumerate(line_string_coords[:-1]):
            target_coords = line_string_coords[ind+1]
            self.add_node(coords, stream_order, target_coords, source_coords, from_tdx)
            source_coords = coords
        self.add_node(target_coords, stream_order, None, source_coords, from_tdx)

    def add_node(self, coordinates, stream_order, target_coordinates, source_coordinates, from_tdx):
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
        coords = node.coordinates
        if len(node.source_nodes) == 0:
            self.source_node_coordinates.add(coords)
        elif len(node.source_nodes) == 1:
            if coords in self.source_node_coordinates:
                self.source_node_coordinates.remove(coords)


class Stream:
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
        if current_node is not None:
            current_node.source_streams.append(self.stream_id)
            current_node.update_stream_order()
            self.from_tdx *= current_node.from_tdx
            self.nodes.append(current_node)
            while len(current_node.source_nodes) == 1 and current_node.target_node is not None:
                current_node = current_node.target_node
                current_node.update_stream_order()
                current_node.source_streams.append(self.stream_id)
                self.from_tdx *= current_node.from_tdx
                self.nodes.append(current_node)

    @cached_property
    def target_stream(self):
        return self.nodes[-1].target_stream

    @cached_property
    def source_streams(self):
        return self.nodes[0].source_streams

    @cached_property
    def line_string(self):
        coords = [node.coordinates for node in self.nodes]
        return shapely.LineString(coords)

class StreamGenerator:
    def __init__(self, node_generator: NodeGenerator):
        self.stream_dict = {}
        self.seen_nodes = set()
        for ind, node_coordinates in enumerate(node_generator.source_node_coordinates):
            self.investigate_node(node_generator.node_info[node_coordinates])
        self.gdf = gpd.GeoDataFrame(
            [
                {
                    'stream_id': stream_id, 'source_streams': stream.source_streams,
                    'target_stream': stream.target_stream, 'stream_order': stream.stream_order,
                    'geometry': stream.line_string
                }
             for stream_id, stream in self.stream_dict.items()
            ], crs=4326
        )

    def investigate_node(self, node: StreamNode):
        if len(node.source_nodes) <= len(node.source_streams) and node.coordinates not in self.seen_nodes:
            self.seen_nodes.add(node.coordinates)
            stream = Stream(node, stream_dict=self.stream_dict)
            if len(stream.nodes) > 1:
                final_node = stream.nodes[-1]
                self.investigate_node(final_node)
            else:
                stream_id = stream.stream_id
                self.stream_dict.pop(stream_id)
