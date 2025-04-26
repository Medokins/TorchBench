import torch.nn as nn
import torch.fx as fx


class ArchitectureComparator:
    def __init__(self, model_a, model_b, method="graph_structural"):
        self.model_a = model_a
        self.model_b = model_b
        self.method = method

    def trace_model_to_graph(self, model, include_params=False):
        traced = fx.symbolic_trace(model)
        nodes = []
        edges = []

        node_ids = {}
        current_id = 0

        for node in traced.graph.nodes:
            if node.op in ['call_module', 'call_function']:
                node_ids[node] = current_id
                if include_params:
                    params = self._extract_params(model, node.target)
                    nodes.append((current_id, (str(node.target), params)))
                else:
                    nodes.append((current_id, str(node.target)))
                current_id += 1

        for node in traced.graph.nodes:
            if node.op in ['call_module', 'call_function']:
                for input_node in node.all_input_nodes:
                    if input_node in node_ids:
                        edges.append((node_ids[input_node], node_ids[node]))

        return nodes, edges

    def _extract_params(self, model, target):
        submodule = dict(model.named_modules()).get(target, None)
        if submodule is None:
            return {}
        if isinstance(submodule, nn.Linear):
            return {'in_features': submodule.in_features, 'out_features': submodule.out_features}
        if isinstance(submodule, nn.Conv2d):
            return {'in_channels': submodule.in_channels, 'out_channels': submodule.out_channels, 'kernel_size': submodule.kernel_size}
        return {}

    def structural_similarity(self):
        nodes_a, edges_a = self.trace_model_to_graph(self.model_a, include_params=False)
        nodes_b, edges_b = self.trace_model_to_graph(self.model_b, include_params=False)

        node_match = self._node_match_score(nodes_a, nodes_b)
        edge_match = self._edge_match_score(edges_a, edges_b)

        similarity = (node_match + edge_match) / 2.0
        return similarity

    def structural_similarity_with_params(self):
        nodes_a, edges_a = self.trace_model_to_graph(self.model_a, include_params=True)
        nodes_b, edges_b = self.trace_model_to_graph(self.model_b, include_params=True)

        node_match = self._node_match_score(nodes_a, nodes_b, with_params=True)
        edge_match = self._edge_match_score(edges_a, edges_b)

        similarity = (node_match + edge_match) / 2.0
        return similarity

    def _node_match_score(self, nodes_a, nodes_b, with_params=False):
        if with_params:
            features_a = [(name, params) for _, (name, params) in nodes_a]
            features_b = [(name, params) for _, (name, params) in nodes_b]
        else:
            features_a = [name for _, name in nodes_a]
            features_b = [name for _, name in nodes_b]

        max_len = max(len(features_a), len(features_b))
        if max_len == 0:
            return 1.0

        matches = 0
        for fa, fb in zip(features_a, features_b):
            if fa == fb:
                matches += 1

        return matches / max_len

    def _edge_match_score(self, edges_a, edges_b):
        max_len = max(len(edges_a), len(edges_b))
        if max_len == 0:
            return 1.0

        matches = 0
        for ea, eb in zip(edges_a, edges_b):
            if ea == eb:
                matches += 1

        return matches / max_len

    def compute_similarity(self):
        if self.method == "graph_structural":
            return self.structural_similarity()
        elif self.method == "graph_structural_with_params":
            return self.structural_similarity_with_params()
        else:
            raise ValueError(f"Unknown comparison method: {self.method}")
