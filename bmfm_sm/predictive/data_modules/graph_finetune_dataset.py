import numpy as np
import torch
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.utils.ndict import NDict
from torch.nn.functional import pad
from torch_geometric.data import Batch

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.fuse_ops.ops_graphs import (
    OpGraphToGraphLaplacian,
    OpSmilesToGraph,
)
from bmfm_sm.predictive.data_modules.mpp_finetune_dataset import MPPFinetuneDataset


class Graph2dFinetuneDataPipeline(MPPFinetuneDataset):
    atom_feature_list = [
        "possible_atomic_num_list",
        "possible_chirality_list",
        "possible_degree_list",
        "possible_formal_charge_list",
        "possible_radical_electron",
        "possible_hybridization_list",
        "possible_IsAromatic_list",
        "possible_numH_list",
        "_ChiralityPossible",
    ]

    bond_feature_list = [
        "possible_bonds",
        "possible_bond_dirs",
        "bond_stereo",
        "bond_is_conjugeated",
        "bond_is_ring",
    ]

    def __init__(self, data_dir, dataset_args, stage="train"):
        super().__init__(data_dir=data_dir, dataset_args=dataset_args, stage=stage)

    def get_ops_pipeline(self) -> PipelineDefault:
        superclass_ops = super().get_ops_pipeline()._ops_and_kwargs

        curr_atom_feature_list = Graph2dFinetuneDataPipeline.atom_feature_list.copy()
        if self.dataset_args.get("include_betti_01", False):
            curr_atom_feature_list.extend(["possible_betti_0", "possible_betti_1"])

        graph_ops = [
            # Generate the Graph object from the SMILES
            (
                OpSmilesToGraph(
                    atom_features=curr_atom_feature_list,
                    bond_features=Graph2dFinetuneDataPipeline.bond_feature_list,
                    embed_node_feats=True,
                    embed_edge_feats=True,
                    fixed_feat_embedding=True,
                    node_feature_type="cate",
                    fixed_feat_embed_categories_max=128,
                ),
                {"key_in": ns.FIELD_SMILES, "key_out": ns.FIELD_GRAPH2D},
            ),
            # Generate eigenvector and values (overwrite the graph Data field)
            (
                OpGraphToGraphLaplacian(),
                {"key_in": ns.FIELD_GRAPH2D, "key_out": ns.FIELD_GRAPH2D},
            ),
        ]

        return PipelineDefault("all_ops", superclass_ops + graph_ops)

    def collate_fn(self, batch):
        data_list = []
        for b in batch:
            # Unpack the sample, set the label as an attribute of the graph (graph.y), convert the x to a long()
            graph, label = b[ns.FIELD_GRAPH2D], b[ns.FIELD_LABEL]
            if graph is not None:
                graph.y = torch.Tensor(np.reshape(label, (1, len(label))))
                graph.x = graph.x.long()
                data_list.append(graph)

        # Get the number of nodes in each graph (and the max num nodes for this batch)
        node_nums = [graph.x.size(0) for graph in data_list]
        edge_nums = [graph.edge_attr.size(0) for graph in data_list]
        max_node_num = max(node_nums)

        # Reorganize data
        basic_tuples = [
            (
                graph.x,  # node_data
                graph.edge_index,
                graph.edge_attr,
                graph.y,
            )
            for graph in data_list
        ]
        # Group together the common attributes
        (
            xs,
            edge_indexs,
            edge_attrs,
            ys,
        ) = zip(*basic_tuples)

        # Initialize collated result
        collated = {}
        y = torch.cat(ys)
        collated["y"] = y
        collated["node_num"] = torch.LongTensor(node_nums)
        collated["node_data"] = torch.cat(xs)
        collated["edge_num"] = torch.LongTensor(edge_nums)
        collated["edge_data"] = torch.cat(edge_attrs).long()
        collated["edge_index"] = torch.cat(edge_indexs, dim=1)

        # Graph Laplacian eigenvectors
        (lap_eigvec, _) = zip(
            *[(graph.lap_eigvec, graph.lap_eigval) for graph in data_list]
        )
        lap_eigvec = torch.cat(
            [
                pad(i, (0, max_node_num - i.size(1)), value=float("0"))
                for i in lap_eigvec
            ]
        )
        collated["lap_eigvec"] = lap_eigvec

        return collated

    # Used for adaptive sampling
    def get_feature_fn(self):
        # num_atoms
        def f(example):
            return example[ns.FIELD_GRAPH2D].x.shape[0]

        return f

    # Helper function - Takes a smile and returns a dictionary with the Graph data
    # Includes all fields expected by the Graph2d Model (node_data, edge_data, laplacian eigenvectors, etc.)
    def smiles_to_graph_format(smiles):
        final_atom_feature_list = Graph2dFinetuneDataPipeline.atom_feature_list.copy()
        op_smiles_to_graph = OpSmilesToGraph(
            atom_features=final_atom_feature_list,
            bond_features=Graph2dFinetuneDataPipeline.bond_feature_list,
            embed_node_feats=True,
            embed_edge_feats=True,
            fixed_feat_embedding=True,
            node_feature_type="cate",
            fixed_feat_embed_categories_max=128,
        )
        op_graph_laplacian = OpGraphToGraphLaplacian()

        # Passing the NDict through the operations
        sample_ndict = NDict({ns.FIELD_SMILES: smiles})
        sample_ndict = op_smiles_to_graph(
            sample_ndict, key_in=ns.FIELD_SMILES, key_out=ns.FIELD_GRAPH2D
        )
        sample_ndict = op_graph_laplacian(
            sample_ndict, key_in=ns.FIELD_GRAPH2D, key_out=ns.FIELD_GRAPH2D
        )

        # Generating/formatting the other metadata needed by the model:
        graph = sample_ndict[ns.FIELD_GRAPH2D]
        graph.x = graph.x.long()
        output_dict = {}
        output_dict["node_data"] = graph.x
        output_dict["node_num"] = torch.LongTensor([graph.x.size(0)])
        output_dict["edge_num"] = torch.LongTensor([graph.edge_attr.size(0)])
        output_dict["edge_data"] = graph.edge_attr.long()
        output_dict["edge_index"] = graph.edge_index
        output_dict["lap_eigvec"] = graph.lap_eigvec

        return output_dict


##########################################################################################
# Similar to the Graph2dFinetuneDataPipeline but will be used by the GCN, GIN, GNN, etc.
##########################################################################################


class Graph2dGNNFinetuneDataPipeline(MPPFinetuneDataset):
    def __init__(self, data_dir, dataset_args, stage="train"):
        super().__init__(data_dir=data_dir, dataset_args=dataset_args, stage=stage)

    def get_ops_pipeline(self) -> PipelineDefault:
        superclass_ops = super().get_ops_pipeline()._ops_and_kwargs

        curr_atom_feature_list = Graph2dFinetuneDataPipeline.atom_feature_list.copy()
        if self.dataset_args.get("include_betti_01", False):
            curr_atom_feature_list.extend(["possible_betti_0", "possible_betti_1"])

        graph_ops = [
            # Generate the Graph object from the SMILES
            (
                OpSmilesToGraph(
                    atom_features=curr_atom_feature_list,
                    bond_features=Graph2dFinetuneDataPipeline.bond_feature_list,
                ),
                {"key_in": ns.FIELD_SMILES, "key_out": ns.FIELD_GRAPH2D_GNN},
            ),
        ]
        return PipelineDefault("all_ops", superclass_ops + graph_ops)

    def collate_fn(self, batch):
        data_list = []
        for b in batch:
            graph, label = b[ns.FIELD_GRAPH2D_GNN], b[ns.FIELD_LABEL]
            if graph is not None:
                np_label = np.reshape(label, (1, len(label)))
                graph.y = torch.from_numpy(np_label)
                data_list.append(graph)

        bd = Batch.from_data_list(data_list)
        bd = bd.contiguous()
        return bd
