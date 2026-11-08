import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn.pytorch import GraphConv
import dgl
from forecast_models.layers.mpts_layers import Inception_CBAM


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        :param x: the input features, shape: (batch_size, n, in_features)
        :param adj: the adjacency matrix, shape: (n, n)
        """
        support = torch.einsum("bnc,ck->bnk", x, self.weight)  # torch.bmm(x, self.weight)
        output = torch.einsum("mn,bnk->bmk", adj, support)  # torch.bmm(adj.unsqueeze(0), support)
        return output


class TemporalConv1D(nn.Module):
    """
    1D Temporal Convolutional Layer for extracting local temporal features
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(TemporalConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                              padding=padding, dilation=dilation)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        """
        :param x: input features, shape: (batch_size, in_channels, seq_len)
        :return: output features, shape: (batch_size, out_channels, output_len)
        """
        return self.conv(x)


class TgrGCN(nn.Module):
    def __init__(self, config, sim_feats, atk_vars, device='cuda'):
        super(TgrGCN, self).__init__()
        self.input_dim = config.bef_tgr_len
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.trigger_len
        self.init_bound = config.epsilon

        self.constant_MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)  # the first element is set to 0
        )
        self.structure_MLP = nn.Sequential(
            nn.Linear(sim_feats.shape[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 64)  # the first element is set to 0
        )

        self.conv1 = GraphConvolutionLayer(self.input_dim, self.hidden_dim)
        self.conv2 = GraphConvolutionLayer(self.hidden_dim, self.output_dim)

        self.sim_feats = torch.from_numpy(sim_feats).float().to(device)[atk_vars]  # (n, c)

        self.device = device
        self.layer_num = 2

        for m in self.constant_MLP:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.uniform_(m.bias, -0.2, 0.2)

        # self.structure_MLP = self.structure_MLP
        # self.constant_MLP = self.constant_MLP

    def forward(self, x, constant_alpha=0.5):
        """
        :param x: the normalized input of the MLP, shape: (batch_size, input_dim)
        """
        n = self.sim_feats.shape[0]
        assert x.shape[0] % n == 0, 'the batch graph size should be a multiple of the number of variables.'
        x = x.view(-1, n, self.input_dim)
        bias = self.constant_MLP(torch.zeros(x.shape[-2], x.shape[-1]).to(self.device))
        bias = torch.tanh(bias) * self.init_bound * constant_alpha

        A = self.cal_structure()
        # symmetric normalization of the adjacency matrix
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A = torch.matmul(torch.matmul(D, A), D)
        A = A.to(self.device)
        h = self.conv1(x, A)
        h = F.relu(h)
        perturb = self.conv2(h, A)

        perturb = torch.tanh(perturb) * self.init_bound * (1 - constant_alpha)
        # add trigger on x[-1] (last element in history) to ensure the real-time property
        out = perturb + bias + x[..., -1:]
        return out, perturb + bias

    def cal_structure(self):
        """
        calculate the similarity matrix of the variables.
        :return: the similarity matrix, shape: (n, n)
        """
        node_num = self.sim_feats.shape[0]
        node_outs = self.structure_MLP(self.sim_feats.detach())  # (n, c)

        # A = torch.matmul(node_outs, node_outs.T)  # (n, n)
        # # print('A shape', A.shape)
        # A = F.tanh(F.relu(A))
        # cosine similarity
        A = F.cosine_similarity(node_outs.unsqueeze(0), node_outs.unsqueeze(1), dim=-1)
        A[A < 0] *= 0
        # add the self-loop
        identity = torch.eye(node_num).to(self.device)
        A = (1 - identity) * A + identity
        return A


class TgrCNN1D(nn.Module):
    """
    1D CNN-based Trigger Generator using local temporal features
    This replaces the graph-based approach with convolutional layers that capture local patterns
    """
    def __init__(self, config, sim_feats, atk_vars, device='cuda', verbose=False):
        super(TgrCNN1D, self).__init__()
        self.verbose = verbose
        #@ Constructor
        self.input_dim = config.bef_tgr_len
        self.hidden_dim = config.Trigger_Config.trigger_generator.hidden_dim
        self.output_dim = config.trigger_len
        self.init_bound = config.epsilon

        # Constant bias MLP (same as original)
        self.constant_MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        # Initial projection from 1 channel to hidden_dim
        self.initial_conv = nn.Conv1d(1, self.hidden_dim, kernel_size=1)
        
        # CNN architecture using Inception_CBAM
        self.cnn = nn.Sequential(
            Inception_CBAM(self.hidden_dim, 1024),
            nn.GELU(),
            Inception_CBAM(1024, self.hidden_dim)
        )
        
        # Final projection to output_dim channels
        self.final_conv = nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=1)
        
        # Adaptive pooling to ensure temporal dimension matches output_dim
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.output_dim)
        
        if sim_feats is not None:
            self.feature_encoder = nn.Sequential(
                nn.Linear(sim_feats.shape[-1], 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
            self.sim_feats = torch.from_numpy(sim_feats).float().to(device)[atk_vars]
        else:
            self.feature_encoder = None
            self.sim_feats = None

        self.device = device

        # Initialize constant MLP
        for m in self.constant_MLP:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.uniform_(m.bias, -0.2, 0.2)

    def forward(self, x, constant_alpha=0.5):
        """
        :param x: the normalized input, shape: (batch_size, input_dim)
        :param constant_alpha: weight for constant bias vs learned perturbation
        :return: (trigger, perturbation) where trigger is the final output and perturbation is the learned part
        """
        # Get number of variables (for batch reshaping if needed)
        if self.verbose:
            print('input x shape', x.shape)
        if self.sim_feats is not None:
            n = self.sim_feats.shape[0]
            if x.shape[0] % n == 0:
                # Reshape to process multiple variables together
                x = x.view(-1, n, self.input_dim)
            else:
                # Single variable per batch
                x = x.unsqueeze(1)  # (batch, 1, input_dim)
                n = 1
        else:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
            n = 1

        # Process input through CNN architecture
        # Reshape x to (batch * n, 1, input_dim) for convolution
        if self.verbose:
            print('x shape', x.shape)
        batch_size = x.shape[0]
        x_conv = x.view(-1, 1, self.input_dim)  # (batch * n, 1, input_dim)
        if self.verbose:
            print('x_conv shape', x_conv.shape)
        
        # Initial projection to hidden_dim channels
        h = self.initial_conv(x_conv)  # (batch * n, hidden_dim, input_dim)
        if self.verbose:
            print('initial_conv h shape', h.shape)
        
        # Pass through Inception_CBAM blocks
        h = self.cnn(h)  # (batch * n, hidden_dim, input_dim)
        if self.verbose:
            print('cnn h shape', h.shape)
        
        # Final projection to output_dim channels
        perturb = self.final_conv(h)  # (batch * n, output_dim, input_dim)
        if self.verbose:
            print('final_conv perturb shape', perturb.shape)
        
        # Adaptive pooling to ensure temporal dimension matches output_dim
        perturb = self.adaptive_pool(perturb)  # (batch * n, output_dim, output_dim)
        if self.verbose:
            print('adaptive_pool perturb shape', perturb.shape)
        
        # Extract diagonal to get final perturbation: for each channel i, take temporal position i
        # This gives us (batch * n, output_dim) - one value per channel
        perturb = torch.diagonal(perturb, dim1=1, dim2=2)  # (batch * n, output_dim)
        if self.verbose:
            print('diagonal perturb shape', perturb.shape)
        
        perturb = torch.tanh(perturb) * self.init_bound * (1 - constant_alpha)
        if self.verbose:
            print('tanh perturb shape', perturb.shape)
        
        # Reshape back to original batch structure
        perturb = perturb.view(batch_size, n, self.output_dim)
        if self.verbose:
            print('reshape perturb shape', perturb.shape)
        
        # Constant bias component (same as original)
        # Use zero input for constant bias generation
        # Generate bias for each variable in the batch
        bias = self.constant_MLP(torch.zeros(batch_size, self.input_dim).to(self.device))
        bias = torch.tanh(bias) * self.init_bound * constant_alpha
        # Reshape bias to match batch structure: (batch_size, n, output_dim)
        bias = bias.unsqueeze(1).expand(-1, n, -1)  # (batch_size, n, output_dim)
        if self.verbose:
            print('bias shape', bias.shape)
        
        # Add trigger on x[-1] (last element in history) to ensure real-time property
        x_last = x[..., -1:]  # (batch, n, 1)
        if self.verbose:
            print('x_last shape', x_last.shape)
        x_last = x_last.expand(-1, -1, self.output_dim)  # (batch, n, output_dim)
        if self.verbose:
            print('x_last shape', x_last.shape)
        
        out = perturb + bias + x_last
        if self.verbose:
            print('out shape', out.shape)
        # Flatten back to (batch * n, output_dim) for consistency
        out = out.view(-1, self.output_dim)
        perturb_flat = perturb.view(-1, self.output_dim)
        bias_flat = bias.reshape(-1, self.output_dim)  # Use reshape instead of view for non-contiguous tensor
        if self.verbose:
            print('out shape', out.shape)
        return out, perturb_flat + bias_flat


class TgrHybridGCN(nn.Module):
    """
    Hybrid Graph Convolution Network that combines features from both TgrGCN and TgrCNN1D
    Uses CNN to extract temporal features and Graph Convolution to incorporate spatial relationships
    """
    def __init__(self, config, sim_feats, atk_vars, device='cuda', verbose=False):
        super(TgrHybridGCN, self).__init__()
        self.verbose = verbose
        self.input_dim = config.bef_tgr_len
        # Use hidden_dim from config if available, otherwise use a default
        if hasattr(config, 'hidden_dim'):
            self.hidden_dim = config.hidden_dim
        elif hasattr(config, 'Trigger_Config') and hasattr(config.Trigger_Config.trigger_generator, 'hidden_dim'):
            self.hidden_dim = config.Trigger_Config.trigger_generator.hidden_dim
        else:
            self.hidden_dim = 256  # default
        self.output_dim = config.trigger_len
        self.init_bound = config.epsilon

        # Constant bias MLP (from both TgrGCN and TgrCNN1D)
        self.constant_MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        # Structure MLP for adjacency matrix computation (from TgrGCN)
        self.structure_MLP = nn.Sequential(
            nn.Linear(sim_feats.shape[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        # CNN feature extraction (from TgrCNN1D)
        # Initial projection from 1 channel to hidden_dim
        self.initial_conv = nn.Conv1d(1, self.hidden_dim, kernel_size=1)
        
        # CNN architecture using Inception_CBAM for temporal feature extraction
        self.cnn = nn.Sequential(
            Inception_CBAM(self.hidden_dim, 1024),
            nn.GELU(),
            Inception_CBAM(1024, self.hidden_dim)
        )
        
        # Final CNN projection to output_dim (for CNN-only perturbation path)
        self.cnn_final_conv = nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=1)
        self.cnn_adaptive_pool = nn.AdaptiveAvgPool1d(self.output_dim)
        
        # Projection from CNN features to graph conv input dimension
        self.cnn_to_gcn = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)

        # Graph Convolution layers (from TgrGCN)
        self.conv1 = GraphConvolutionLayer(self.hidden_dim, self.hidden_dim)
        self.conv2 = GraphConvolutionLayer(self.hidden_dim, self.output_dim)

        # Similarity features for adjacency matrix
        self.sim_feats = torch.from_numpy(sim_feats).float().to(device)[atk_vars]  # (n, c)

        self.device = device

        # Initialize constant MLP
        for m in self.constant_MLP:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.uniform_(m.bias, -0.2, 0.2)

    def forward(self, x, constant_alpha=0.5, feature_alpha=0.5):
        """
        :param x: the normalized input, shape: (batch_size, input_dim)
        :param constant_alpha: weight for constant bias vs learned perturbation
        :param feature_alpha: scale factor to control the magnitude of CNN features vs GCN features
                              (0.0 = pure GCN, 1.0 = pure CNN, 0.5 = equal blend)
        :return: (trigger, perturbation) where trigger is the final output and perturbation is the learned part
        """
        n = self.sim_feats.shape[0]
        if self.verbose:
            print('input x shape', x.shape)
        
        # Reshape input for batch processing
        if x.shape[0] % n == 0:
            x = x.view(-1, n, self.input_dim)  # (batch_size, n, input_dim)
        else:
            # Single variable per batch
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
            n = 1
        
        batch_size = x.shape[0]
        if self.verbose:
            print('reshaped x shape', x.shape)

        # Step 1: Extract temporal features using CNN (from TgrCNN1D approach)
        # Reshape x to (batch * n, 1, input_dim) for convolution
        x_conv = x.view(-1, 1, self.input_dim)  # (batch * n, 1, input_dim)
        if self.verbose:
            print('x_conv shape', x_conv.shape)
        
        # Initial projection to hidden_dim channels
        h_cnn = self.initial_conv(x_conv)  # (batch * n, hidden_dim, input_dim)
        if self.verbose:
            print('initial_conv h_cnn shape', h_cnn.shape)
        
        # Pass through Inception_CBAM blocks for temporal feature extraction
        h_cnn = self.cnn(h_cnn)  # (batch * n, hidden_dim, input_dim)
        if self.verbose:
            print('cnn h_cnn shape', h_cnn.shape)
        
        # CNN-only perturbation path (from TgrCNN1D approach)
        cnn_perturb_raw = self.cnn_final_conv(h_cnn)  # (batch * n, output_dim, input_dim)
        cnn_perturb_raw = self.cnn_adaptive_pool(cnn_perturb_raw)  # (batch * n, output_dim, output_dim)
        cnn_perturb_raw = torch.diagonal(cnn_perturb_raw, dim1=1, dim2=2)  # (batch * n, output_dim)
        cnn_perturb_raw = cnn_perturb_raw.view(batch_size, n, self.output_dim)  # (batch_size, n, output_dim)
        cnn_perturb = torch.tanh(cnn_perturb_raw) * self.init_bound * (1 - constant_alpha)
        if self.verbose:
            print('cnn_perturb shape', cnn_perturb.shape)
        
        # Project CNN features to graph conv input dimension for GCN path
        h_cnn_gcn = self.cnn_to_gcn(h_cnn)  # (batch * n, hidden_dim, input_dim)
        if self.verbose:
            print('cnn_to_gcn h_cnn shape', h_cnn_gcn.shape)
        
        # Extract temporal features: use mean across temporal dimension
        # This gives us (batch * n, hidden_dim) - one feature vector per sample
        h_temporal = h_cnn_gcn.mean(dim=2)  # (batch * n, hidden_dim)
        if self.verbose:
            print('h_temporal shape', h_temporal.shape)
        
        # Reshape back to (batch_size, n, hidden_dim) for graph convolution
        h_temporal = h_temporal.view(batch_size, n, self.hidden_dim)
        if self.verbose:
            print('h_temporal reshaped shape', h_temporal.shape)

        # Step 2: Apply graph convolution on temporal features (from TgrGCN approach)
        A = self.cal_structure()
        # Symmetric normalization of the adjacency matrix
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A = torch.matmul(torch.matmul(D, A), D)
        A = A.to(self.device)
        if self.verbose:
            print('adjacency matrix A shape', A.shape)
        
        # First graph convolution layer
        h_gcn = self.conv1(h_temporal, A)  # (batch_size, n, hidden_dim)
        h_gcn = F.relu(h_gcn)
        if self.verbose:
            print('conv1 h_gcn shape', h_gcn.shape)
        
        # Second graph convolution layer to get final perturbation
        gcn_perturb_raw = self.conv2(h_gcn, A)  # (batch_size, n, output_dim)
        if self.verbose:
            print('conv2 gcn_perturb_raw shape', gcn_perturb_raw.shape)
        
        gcn_perturb = torch.tanh(gcn_perturb_raw) * self.init_bound * (1 - constant_alpha)
        if self.verbose:
            print('gcn_perturb shape', gcn_perturb.shape)
        
        # Step 3: Blend CNN and GCN perturbations using feature_alpha
        # feature_alpha controls the magnitude: alpha * CNN + (1 - alpha) * GCN
        perturb = feature_alpha * cnn_perturb + (1 - feature_alpha) * gcn_perturb
        if self.verbose:
            print('blended perturb shape', perturb.shape)

        # Step 4: Generate constant bias (from both approaches)
        bias = self.constant_MLP(torch.zeros(batch_size, self.input_dim).to(self.device))
        bias = torch.tanh(bias) * self.init_bound * constant_alpha
        # Reshape bias to match batch structure: (batch_size, n, output_dim)
        bias = bias.unsqueeze(1).expand(-1, n, -1)  # (batch_size, n, output_dim)
        if self.verbose:
            print('bias shape', bias.shape)
        
        # Step 5: Combine all components
        # Add trigger on x[-1] (last element in history) to ensure real-time property
        x_last = x[..., -1:]  # (batch, n, 1)
        x_last = x_last.expand(-1, -1, self.output_dim)  # (batch, n, output_dim)
        if self.verbose:
            print('x_last shape', x_last.shape)
        
        out = perturb + bias + x_last
        if self.verbose:
            print('out shape', out.shape)
        
        # Flatten back to (batch * n, output_dim) for consistency
        out = out.view(-1, self.output_dim)
        perturb_flat = perturb.view(-1, self.output_dim)
        bias_flat = bias.reshape(-1, self.output_dim)
        if self.verbose:
            print('final out shape', out.shape)
        
        return out, perturb_flat + bias_flat

    def cal_structure(self):
        """
        Calculate the similarity matrix of the variables (from TgrGCN).
        :return: the similarity matrix, shape: (n, n)
        """
        node_num = self.sim_feats.shape[0]
        node_outs = self.structure_MLP(self.sim_feats.detach())  # (n, c)

        # Cosine similarity
        A = F.cosine_similarity(node_outs.unsqueeze(0), node_outs.unsqueeze(1), dim=-1)
        A[A < 0] *= 0
        # Add the self-loop
        identity = torch.eye(node_num).to(self.device)
        A = (1 - identity) * A + identity
        return A