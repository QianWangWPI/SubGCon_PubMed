import torch
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, to_hetero,HGTConv
from torch import nn, optim
import torch.nn.functional as F
from torch import Tensor





class homoGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x




class NIH(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, data=None):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=0.5)  # 添加 Dropout，50% 的概率丢弃
        for _ in range(num_layers):
            conv = HeteroConv({
                ('pi', 'IsMappedTo', 'author'): SAGEConv((-1, -1), hidden_channels),
                ('author', 'rev_IsMappedTo', 'pi'): SAGEConv((-1, -1), hidden_channels),
                ('author', 'writes', 'pub'): SAGEConv((-1, -1), hidden_channels),
                ('pub', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_channels),

                ('pub', 'cite', 'pub'): GCNConv(-1, hidden_channels),

                # ('affiliation', 'has', 'author'): SAGEConv((-1, -1), hidden_channels),
                # ('venue', 'host', 'pub'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        self.x_feature_len = data['pi'].x.shape[1]
        self.lin = Linear(hidden_channels+self.x_feature_len, out_channels)

    def forward(self, x_dict, edge_index_dict):
        original_pi_features = x_dict['pi']

        for conv in self.convs:
            # original_pi_features = x_dict['pi']
            x_dict = conv(x_dict, edge_index_dict)
            # updated_pi_features = x_dict['pi']
            # concatenated_features = torch.cat([original_pi_features, updated_pi_features], dim=1)
            # x_dict['pi'] = concatenated_features
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        updated_pi_features = x_dict['pi']
        concatenated_features = torch.cat([original_pi_features, updated_pi_features], dim=1)
        logits = self.lin(concatenated_features)
        # return self.lin(x_dict['pi']), x_dict
        return F.softmax(logits),x_dict


class DBLP(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        # 创建一个用于保存每一层卷积操作的模块列表
        self.convs = torch.nn.ModuleList()

        # 定义多层卷积
        for _ in range(num_layers):
            conv = HeteroConv({
                ('author', 'to', 'paper'): SAGEConv((-1, -1), hidden_channels),
                ('paper', 'to', 'author'): SAGEConv((-1, -1), hidden_channels),
                ('paper', 'to', 'term'): SAGEConv((-1, -1), hidden_channels),
                ('term', 'to', 'paper'): SAGEConv((-1, -1), hidden_channels),
                # ('paper', 'to', 'conference'): SAGEConv((-1, -1), hidden_channels),
                # ('conference', 'to', 'paper'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        # 最后的线性层，用于节点分类 (例如，预测 author 的类别)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 遍历每一层卷积
        for conv in self.convs:
            # 对每个关系类型进行卷积
            x_dict = conv(x_dict, edge_index_dict)
            # 对卷积后的特征应用 ReLU 激活函数
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            # logits = self.lin(x_dict['author'])
        # 返回对 'author' 节点的特征进行线性变换后的结果
        return self.lin(x_dict['author']), x_dict

class OGB_MAG(torch.nn.Module):
    # def __init__(self, hidden_channels, out_channels, num_layers):
    #     super().__init__()
    #
    #     self.convs = torch.nn.ModuleList()
    #     for _ in range(num_layers):
    #         conv = HeteroConv({
    #
    #             ('author', 'affiliated_with', 'institution'): SAGEConv((-1, -1), hidden_channels),
    #             ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
    #             ('paper', 'cites', 'paper'): SAGEConv((-1, -1), hidden_channels),
    #             ('paper', 'has_topic', 'field_of_study'): SAGEConv((-1, -1), hidden_channels),
    #         }, aggr='sum')
    #         self.convs.append(conv)
    #
    #     self.lin = Linear(hidden_channels, out_channels)
    #
    # def forward(self, x_dict, edge_index_dict):
    #     for conv in self.convs:
    #         x_dict = conv(x_dict, edge_index_dict)
    #
    #         x_dict = {key: F.relu(x) for key, x in x_dict.items()}
    #     return self.lin(x_dict['paper']),x_dict
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers,data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author']),x_dict

class SubGCon(torch.nn.Module):
    def __init__(self, model, hidden_channels, out_channels, num_layers, data):
        super().__init__()
        self.model = model
        self.graph_temperature_scale = NIH(hidden_channels, out_channels, num_layers, data)

        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=0.5)  # 添加 Dropout，50% 的概率丢弃
        for _ in range(num_layers):
            conv = HeteroConv({
                # ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
                ('pi', 'IsMappedTo', 'author'): SAGEConv((-1, -1), hidden_channels),
                ('author', 'rev_IsMappedTo', 'pi'): SAGEConv((-1, -1), hidden_channels),
                # ('paper', 'rev_writes', 'author'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
                ('paper', 'writeBy', 'author'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(64, 10)
        self.lin3 = Linear(out_channels, 1)
    def forward(self, x_dict, edge_index_dict):
        logits,_ = self.model(x_dict, edge_index_dict)
        _['pi'] = self.lin2(_['pi'])
        ll1,_ = self.graph_temperature_scale(_,edge_index_dict)
        temperature = self.lin3(ll1)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return logits/temperature,x_dict
        # return self.lin(x_dict['pi'])


class SubGCon2(torch.nn.Module):
    def __init__(self, model, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.model = model
        self.graph_temperature_scale = DBLP(hidden_channels, out_channels, num_layers)

        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=0.5)  # 添加 Dropout，50% 的概率丢弃
        for _ in range(num_layers):
            conv = HeteroConv({
                ('author', 'to', 'paper'): SAGEConv((-1, -1), hidden_channels),
                ('paper', 'to', 'author'): SAGEConv((-1, -1), hidden_channels),
                ('paper', 'to', 'term'): SAGEConv((-1, -1), hidden_channels),
                ('term', 'to', 'paper'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

        self.lin2 = Linear(out_channels, 1)
    def forward(self, x_dict, edge_index_dict):
        logits,_ = self.model(x_dict, edge_index_dict)
        ll1,_ = self.graph_temperature_scale(_,edge_index_dict)
        temperature = self.lin2(ll1)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return logits/temperature,x_dict



class TS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index):
        logits,_ = self.model(x, edge_index)
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def temperature_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return temperature