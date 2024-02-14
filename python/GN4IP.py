# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:05:37 2024

@author: Billy_Herzberg

This contains the new and improved version of the GN4IP package that can be
used to create, train, and test graph neural networks for image reconstruction.
Analogous convolutional neural networks can also be made, trained, and tested.
"""

from time import time
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Conv3d, ConvTranspose3d, MaxPool3d
import torch_geometric
from torch_geometric.nn import GCNConv, max_pool_x


def simulateGraphData(n_samples, n_nodes, n_edges, n_features=1, n_pooling_layers=0, dtype=torch.double):
    """
    Simulate some random graph data that can be used to test some functions

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_nodes : int
        Number of nodes in the input graph.
    n_edges : int
        Number of edges in the input graph.
    n_features : int
        Number of features on each node of the input graph.
    n_pooling_layers : int
        Number of pooling layers to account for (clusters and edges).
    dtype : torch.dtype
        The type of the data

    Returns
    -------
    x : torch.Tensor (n_samples, n_nodes, n_features)
        Graph feature matrix.
    edge_index_list : list of torch.Tensor (2, n_edges)
        List of edge_index tensors defining the edges in the graph.
    clusters_list : list of torch.Tensor (n_nodes)
        List of cluster assignment tensors.

    """
    x = torch.rand((n_samples, n_nodes, n_features), dtype=dtype)
    edge_index = torch.randint(low=0, high=n_nodes, size=(2, n_edges))
    if n_edges > n_nodes:
        #print("n_nodes:", n_nodes)
        edge_index[0, 0:n_nodes] = torch.arange(n_nodes)
    edge_index_list = [edge_index]
    clusters_list = []
    
    # At each depth, make random clusters and random new edges
    for _ in range(n_pooling_layers):
        n_clusters = n_nodes // 4
        clusters = torch.randint(low=0, high=n_clusters, size=(n_nodes,))
        clusters[0:n_clusters] = torch.arange(n_clusters)
        clusters_list.append(clusters)
        n_nodes = n_clusters
        n_edges = n_edges // 4
        edge_index = torch.randint(low=0, high=n_nodes, size=(2, n_edges))
        if n_edges > n_nodes:
            #print("n_nodes:", n_nodes)
            edge_index[0, 0:n_nodes] = torch.arange(n_nodes)
        edge_index_list.append(edge_index)
        
    return x, edge_index_list, clusters_list

def simulateGridData(n_samples, n_features, height, width, depth=None, dtype=torch.double):
    """
    Simulate some random grid data that can be used to test some functions

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features on each pixel.
    width : int
        Number of pixels in the width dimension.
    height : int
        Number of pixels in the height dimension.
    depth : int | None
        Number of pixels in the depth dimension (for 3d images) or None for 2d.
    dtype : torch.dtype
        The type of the data

    Returns
    -------
    x : Tensor
        A torch.Tensor with shape (n_samples, n_features, height, width)

    """
    if depth is None:
        x = torch.rand((n_samples, n_features, height, width), dtype=dtype)
    else:
        x = torch.rand((n_samples, n_features, depth, height, width), dtype=dtype)
    return x


class GraphUNet(torch.nn.Module):
    
    def __init__(self, channels_in=1, channels=2, channels_out=1, convolutions=1, n_pooling_layers=0):
        """
        Initialize the graph u-net model and define the layers.

        Parameters
        ----------
        channels_in : int
            Number of input channels to the network.
        channels : int
            Number of output channels to the first convolution.
        channels_out : int
            Number of output channels.
        convolutions : int
            Number of convolutions at each depth.
        n_pooling_layers : int
            Number of pooling (and unpooling) layers.

        Returns
        -------
        None.

        """
        super(GraphUNet, self).__init__()
        self.type = "gnn"
        self.channels_in = channels_in
        self.channels = channels
        self.channels_out = channels_out
        self.convolutions = convolutions
        self.n_pooling_layers = n_pooling_layers
        self.buildLayers()
        self.double()
        self.resetParameters()
        
    def buildLayers(self):
        """
        Define the layers of the network

        Returns
        -------
        None.

        """
        # Define the initial convolution block
        chan_in = self.channels_in
        chan_out = self.channels
        self.first_convs = self.buildGCNBlock(chan_in, chan_out)
        
        # Define the encoder side of the u-net
        self.encoder = torch.nn.ModuleList()
        for i in range(self.n_pooling_layers):
            chan_in = chan_out
            chan_out = chan_out * 2
            self.encoder.append(self.buildGCNBlock(chan_in, chan_out))
        
        # Define the decoder side of the u-net
        self.decoder = torch.nn.ModuleList()
        for i in range(self.n_pooling_layers):
            chan_in = chan_out
            chan_out = chan_in // 2
            self.decoder.append(self.buildGCNBlock(chan_in+chan_out, chan_out))
        
        # Define the final output layer
        self.final_layer = GCNConv(chan_out, self.channels_out)
    
    def buildGCNBlock(self, chan_in, chan_out):
        """
        Define a graph convolutional block as a series of convolutions

        Parameters
        ----------
        chan_in : int
            Number of input features.
        chan_out : int
            Number of output features.

        Returns
        -------
        block : torch.nn.ModuleList
            A list of graph convolutional layers.

        """
        block = torch.nn.ModuleList()
        for i in range(self.convolutions):
            block.append(GCNConv(chan_in, chan_out))
            chan_in = chan_out
        return block

    def resetParameters(self):
        """
        Loop through all convolution ModuleLists and reset the parameters

        Returns
        -------
        None.

        """
        for conv in self.first_convs:
            conv.reset_parameters()
        for block in self.encoder:
            for conv in block:
                conv.reset_parameters()
        for block in self.decoder:
            for conv in block:
                conv.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, x, edge_index_list, clusters_list=[], batch=None):
        """
        Pass data (x, edge_index) through the model. Careful about batch.

        Parameters
        ----------
        x : torch.Tensor
            The feature matrix of the input graph.
        edge_index_list : list of torch.Tensor
            List of tensors containing edge information at each depth.
        clusters_list : list of torch.Tensor
            List of tensors containing cluster information for each pooling.
        batch : torch.Tensor | None
            Defines which graph the nodes in x belong to (for pooling).

        Returns
        -------
        x : torch.Tensor
            The feature matrix of the output graph.

        """
        
        # Initialize list for storing skip connection info
        xs = []
        
        # Do the first convolution block
        #print(x.type())
        x = self.doGCNBlock(x, edge_index_list[0], self.first_convs)
        #print(x.size())
        
        # Do the encoder side with pooling and convolutions
        for i in range(self.n_pooling_layers):
            xs += [x]
            #print(i, x.size(), clusters_list[i].size(), max(clusters_list[i]))
            x, batch = max_pool_x(clusters_list[i], x, batch)
            #print(x.size())
            x = self.doGCNBlock(x, edge_index_list[i+1], self.encoder[i])
            #print(x.type())
        
        # Do the decoder side with unpooling and convolutions
        for i in range(self.n_pooling_layers):
            j = self.n_pooling_layers - i - 1
            x = torch.index_select(x, 0, clusters_list[j])
            #print(x.size())
            x = torch.cat((xs[j], x,), dim=1)
            #print(x.size())
            x = self.doGCNBlock(x, edge_index_list[j], self.decoder[i])
            #print(x.type())
        
        # Do the final convolution without edges and return
        edge_index = torch.empty((2,0), dtype=edge_index_list[0].dtype, device=edge_index_list[0].device)
        x = self.final_layer(x, edge_index)
        #print(x.type())
        return x
    
    def doGCNBlock(self, x, edge_index, block):
        """
        Pass the graph (x, edge_index) through the block

        Parameters
        ----------
        x : torch.Tensor
            The feature matrix of the input graph.
        edge_index : torch.Tensor
            Tensor containing edge information of the input graph.
        block : torch.nn.ModuleList
            A list of graph convolutional layers.

        Returns
        -------
        x : torch.Tensor
            The feature matrix of the output graph.

        """
        for conv in block:
            #print(x.size(), torch.max(edge_index))
            x = conv(x, edge_index)
            x = F.relu(x)
        return x
    
    def getDevice(self):
        """
        Return the device that the model is on

        Returns
        -------
        torch.device
            Where the model is currently (based on the parameters location).

        """
        return next(iter(self.state_dict().items()))[1].device

    def fit(self, data_tr, data_va, params_tr):
        """
        Fit the model to the training data

        Parameters
        ----------
        data_tr : tuple of (x, y, edge_index_list, clusters_list)
            Data to use for training the model.
        data_va : tuple of (x, y, edge_index_list, clusters_list)
            Data to use for validating the model.
        params_tr : dict
            Dictionary of training parameters such as batch size.
            -> batch_size : int
            -> device : torch.Device
            -> learning_rate : float
            -> max_epochs : int
            -> loss_function : func

        Returns
        -------
        training_output : dict
            Dictionary containing output from training such as losses.

        """
        training_output = fitModel(self, data_tr, data_va, params_tr)
        return training_output
    
    def predict(self, data_pr, params_pr):
        """
        Use the model to make predictions

        Parameters
        ----------
        data_pr : tuple of (x, y, edge_index_list, clusters_list)
            Data to make predictions on using the model
        params_pr : dict
            Dictionary of prediction parameters such as device.
            -> device : torch.Device
            -> loss_function : func

        Returns
        -------
        predict_output : dict
            Dictionary containing output from the prediction.

        """
        predict_output = predictModel(self, data_pr, params_pr)
        return predict_output
    
class ConvUNet(torch.nn.Module):
    
    def __init__(self, dim=2, channels_in=1, channels=2, channels_out=1, convolutions=1, n_pooling_layers=0, kernel_size=3, stride=1, padding="same"):
        """
        Parameters
        ----------
        Initialize the convolutional (cnn) u-net model and define the layers.

        Parameters
        ----------
        dim : int (2 | 3)
            Dimension of the data
        channels_in : int
            Number of input channels to the network.
        channels : int
            Number of output channels to the first convolution.
        channels_out : int
            Number of output channels.
        convolutions : int
            Number of convolutions at each depth.
        n_pooling_layers : int
            Number of pooling (and unpooling) layers.
        kernel_size : int
            Size of the convolutional kernels for the Conv2d layers
        stride : int
            Stride for the Conv2d layers
        padding : int | string
            Padding for the Conv2d layers. "same" or "valid" are string options

        Returns
        -------
        None.

        """
        super(ConvUNet, self).__init__()
        self.type = "cnn"
        self.dim = dim
        self.channels_in = channels_in
        self.channels = channels
        self.channels_out = channels_out
        self.convolutions = convolutions
        self.n_pooling_layers = n_pooling_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.buildLayers()
        self.double()
        self.resetParameters()
        
    def buildLayers(self):
        """
        Define the layers of the network

        Returns
        -------
        None.

        """
        # Define the initial convolution block
        chan_in = self.channels_in
        chan_out = self.channels
        self.first_convs = self.buildConvBlock(chan_in, chan_out)
        
        # Define the encoder side of the u-net
        self.encoder = torch.nn.ModuleList()
        for i in range(self.n_pooling_layers):
            chan_in = chan_out
            chan_out = chan_out * 2
            self.encoder.append(self.buildConvBlock(chan_in, chan_out))
        if self.dim == 2:
            self.pool_layer = MaxPool2d(kernel_size=2, stride=2)
        elif self.dim == 3:
            self.pool_layer = MaxPool3d(kernel_size=2, stride=2)
        
        # Define the decoder side of the u-net
        self.decoder = torch.nn.ModuleList()
        self.tr_convs = torch.nn.ModuleList()
        for i in range(self.n_pooling_layers):
            chan_in = chan_out
            chan_out = chan_in // 2
            if self.dim == 2:
                self.tr_convs.append(ConvTranspose2d(chan_in, chan_out, kernel_size=2, stride=2))
            elif self.dim == 3:
                self.tr_convs.append(ConvTranspose3d(chan_in, chan_out, kernel_size=2, stride=2))
            self.decoder.append(self.buildConvBlock(chan_in, chan_out))
        
        # Define the final output layer
        if self.dim == 2:
            self.final_layer = Conv2d(chan_out, self.channels_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        elif self.dim == 3:
            self.final_layer = Conv3d(chan_out, self.channels_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
    def buildConvBlock(self, chan_in, chan_out):
        """
        Define a convolutional block as a series of convolutions

        Parameters
        ----------
        chan_in : int
            Number of input features.
        chan_out : int
            Number of output features.

        Returns
        -------
        block : torch.nn.ModuleList
            A list of convolutional layers.

        """
        block = torch.nn.ModuleList()
        if self.dim == 2:
            for i in range(self.convolutions):
                block.append(Conv2d(chan_in, chan_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
                chan_in = chan_out
        elif self.dim == 3:
            for i in range(self.convolutions):
                block.append(Conv3d(chan_in, chan_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
                chan_in = chan_out
        else:
            raise Exception("The ConvUNet dim must be either 2 or 3, not {}".format(self.dim))
            
        return block
    
    def resetParameters(self):
        """
        Loop through all convolution ModuleLists and reset the parameters

        Returns
        -------
        None.

        """
        for conv in self.first_convs:
            conv.reset_parameters()
        for block in self.encoder:
            for conv in block:
                conv.reset_parameters()
        for block in self.decoder:
            for conv in block:
                conv.reset_parameters()
        for conv in self.tr_convs:
            conv.reset_parameters()
        self.final_layer.reset_parameters()
    
    def forward(self, x):
        """
        Pass data (x) through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        x : torch.Tensor
            The output data.

        """
        
        # Initialize list for storing skip connection info
        xs = []
        
        # Do the first convolution block
        #print(x.size())
        x = self.doConvBlock(x, self.first_convs)
        #print(x.size())
        
        # Do the encoder side with pooling and convolutions
        for i in range(self.n_pooling_layers):
            xs += [x]
            x = self.pool_layer(x)
            x = self.doConvBlock(x, self.encoder[i])
            #print(x.size())
        
        # Do the decoder side with unpooling and convolutions
        for i in range(self.n_pooling_layers):
            j = self.n_pooling_layers - i - 1
            x = self.tr_convs[i](x)
            #print(x.size())
            x = torch.cat((xs[j], x,), dim=1)
            x = self.doConvBlock(x, self.decoder[i])
            #print(x.size())
        
        # Do the final convolution and return
        x = self.final_layer(x)
        #print(x.size())
        return x
    
    def doConvBlock(self, x, block):
        """
        Pass the data (x) through the block

        Parameters
        ----------
        x : torch.Tensor
            The input data.
        block : torch.nn.ModuleList
            A list of convolutional layers.

        Returns
        -------
        x : torch.Tensor
            The output data.

        """
        for conv in block:
            #print(x.size())
            x = conv(x)
            x = F.relu(x)
        return x
    
    def getDevice(self):
        """
        Return the device that the model is on

        Returns
        -------
        torch.device
            Where the model is currently (based on the parameters location).

        """
        return next(iter(self.state_dict().items()))[1].device

    def fit(self, data_tr, data_va, params_tr):
        """
        Fit the model to the training data
    
        Parameters
        ----------
        data_tr : tuple of (x, y)
            Data to use for training the model.
        data_va : tuple of (x, y)
            Data to use for validating the model.
        params_tr : dict
            Dictionary of training parameters such as batch size.
            -> batch_size : int
            -> device : torch.Device
            -> learning_rate : float
            -> max_epochs : int
            -> loss_function : func
    
        Returns
        -------
        training_output : dict
            Dictionary containing output from training such as losses.
    
        """
        training_output = fitModel(self, data_tr, data_va, params_tr)
        return training_output
    
    def predict(self, data_pr, params_pr):
        """
        Use the model to make predictions

        Parameters
        ----------
        data_pr : tuple of (x, y)
            Data to make predictions on using the model
        params_pr : dict
            Dictionary of prediction parameters such as device.
            -> device : torch.Device
            -> loss_function : func

        Returns
        -------
        predict_output : dict
            Dictionary containing output from the prediction.

        """
        predict_output = predictModel(self, data_pr, params_pr)
        return predict_output
    
def fitModel(model, data_tr, data_va, params_tr):
    """
    Fit the neural network model to the training data

    Parameters
    ----------
    model : torch.nn.Module (typically GraphUNet or ConvUNet)
        The model to fit the parameters of.
    data_tr : tuple of (x, y, {edge_index_list}, {clusters_list})
        Data to use for training the model.
    data_va : tuple of (x, y, {edge_index_list}, {clusters_list})
        Data to use for validating the model.
    params_tr : dict
        Dictionary of training parameters such as batch size.
        -> batch_size : int
        -> device : torch.Device
        -> learning_rate : float
        -> max_epochs : int
        -> loss_function : func
        -> patience : int
        -> print_freq : int

    Returns
    -------
    training_output : dict
        Dictionary containing output from training such as losses.
    """
    # Check that the model.type is "gnn" or "cnn"
    if not (model.type=="gnn" or model.type=="cnn"):
        raise Exception("Need model.type to be one of { gnn | cnn }")
    
    # Create the training and validation loaders
    if model.type == "gnn":
        loader_tr = loaderGNN(data_tr, params_tr["batch_size"], shuffle=True)
        loader_va = loaderGNN(data_va)
    elif model.type == "cnn":
        loader_tr = loaderCNN(data_tr, params_tr["batch_size"], shuffle=True)
        loader_va = loaderCNN(data_va)
    
    # Move the model to the training device
    model.to(params_tr["device"])
    
    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = params_tr["learning_rate"])
    
    # Prepare for the main loop
    loss_tr = torch.empty((0,))
    loss_va = torch.empty((0,))
    
    # Setup early stopping based on loss_va
    state_dict_best = None
    min_loss = 10**6
    patience = params_tr["patience"]
    patience_reset = params_tr["patience"]
    
    # Start the main loop
    if params_tr["print_freq"] > 0:
        print("Training a GNN...")
        print("(  time  ) epoch :  loss tr  :  loss va ")
        print("----------------------------------------")
    t_start = time()
    for epoch in range(1, params_tr["max_epochs"]+1):
        if model.type == "gnn":
            loss_tr = torch.cat((loss_tr, trainEpochGNN(model, loader_tr, data_tr[2], data_tr[3], optimizer, params_tr["loss_function"]).unsqueeze(0)))
            loss_va = torch.cat((loss_va, validEpochGNN(model, loader_va, data_va[2], data_va[3], params_tr["loss_function"]).unsqueeze(0)))
        elif model.type == "cnn":
            loss_tr = torch.cat((loss_tr, trainEpochCNN(model, loader_tr, optimizer, params_tr["loss_function"]).unsqueeze(0)))
            loss_va = torch.cat((loss_va, validEpochCNN(model, loader_va, params_tr["loss_function"]).unsqueeze(0)))
            
        # Print a training update
        if params_tr["print_freq"] > 0 and (epoch % params_tr["print_freq"] == 0 or epoch == 1):
            t = time() - t_start
            h = int(t // 3600)
            m = int((t - h*3600) // 60)
            s = (t - h*3600 - m*60)
            print("({:02d}:{:02d}:{:02.0f}) {:4d}  : {:8.5f}  : {:8.5f}".format(h, m, s, epoch, loss_tr[-1], loss_va[-1]))
            
        # Check for early stopping
        if loss_va[-1] < min_loss:
            min_loss = loss_va[-1].item()
            state_dict_best = deepcopy(model.state_dict())
            patience = patience_reset
        else:
            patience += -1
            if patience <= 0:
                model.load_state_dict(state_dict_best)
                print("-> Stopped early (epoch {}) and best parameters loaded back in.".format(epoch))
                break
            
    # Move the model back to the cpu
    model.cpu()
    
    training_output = {
        "training_time" : time() - t_start,
        "loss_tr" : loss_tr,
        "loss_va" : loss_va
    }
    return training_output

def loaderGNN(data, batch_size=1, shuffle=False):
    """
    Create a loader to feed data through a model

    Parameters
    ----------
    data : tuple of (x, y) (each are size (n_samples, n_nodes, n_features))
        Data to use for training the model as tensors.
    batch_size : int
        Size of each batch.
    shuffle : bool
        Whether to shuffle the batches when looping

    Returns
    -------
    loader : torch_geometric.loader.dataloader.DataLoader
        A loader to use in training or validation.

    """
    dataset = []
    for i in range(data[0].size(0)):
        dataset.append(torch_geometric.data.Data(x=data[0][i, :, :], y=data[1][i, :, :]))
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def loaderCNN(data, batch_size=1, shuffle=False):
    """
    Create a loader to feed data through a model

    Parameters
    ----------
    data : tuple of (x, y) (each are size n_samples, n_features, (d), h, w)
        Data to use for training the model as tensors.
    batch_size : int
        Size of each batch.
    shuffle : bool
        Whether to shuffle the batches when looping

    Returns
    -------
    loader : torch.utils.data.DataLoader
        A loader to use in training or validation.

    """
    dataset = torch.utils.data.TensorDataset(data[0], data[1])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)    
    return loader

def trainEpochGNN(model, loader_tr, edge_index_list, clusters_list, optimizer, loss_function):
    """
    Execute one epoch of training the model
    
    Parameters
    ----------
    model : GraphUNet
        The model to fit the parameters of.
    loader_tr : torch_geometric.loader.dataloader.DataLoader
        A loader to use in training or validation.
    edge_index_list : list of torch.Tensor
        List of tensors containing edge information at each depth.
    clusters_list : list of torch.Tensor
        List of tensors containing cluster information for each pooling.
    optimizer : torch.optim.Optimizer
        Parameter optimization routine.
    loss_function : func
        Function for computing the loss.

    Returns
    -------
    float
        loss value for the model and loader.

    """
    # Move some things to the model's device
    device = model.getDevice()
    edge_index_list = [ei.to(device) for ei in edge_index_list]
    clusters_list = [cl.to(device) for cl in clusters_list]
    
    model.train()
    n_samples = 0
    loss_total = 0
    for batch in loader_tr:
        # Move some things to the model's device
        x = batch.x.to(device)
        y = batch.y.to(device)
        b = batch.batch.to(device)
        optimizer.zero_grad()
        ei_list, cl_list = fixEdgesAndClustersForBatch(edge_index_list, clusters_list, b)
        yhat = model(x, ei_list, cl_list, b)
        loss = loss_function(yhat, y)
        loss.backward()
        optimizer.step()
        n_samples += (b[-1] + 1)
        loss_total += (loss * (b[-1] + 1))
    return (loss_total / n_samples).cpu().detach()

def validEpochGNN(model, loader_va, edge_index_list, clusters_list, loss_function):
    """
    Execute one epoch of validating the model
    
    Parameters
    ----------
    model : GraphUNet
        The model to fit the parameters of.
    loader_va : torch_geometric.loader.dataloader.DataLoader
        A loader to use in training or validation.
    edge_index_list : list of torch.Tensor
        List of tensors containing edge information at each depth.
    clusters_list : list of torch.Tensor
        List of tensors containing cluster information for each pooling.
    loss_function : func
        Function for computing the loss.

    Returns
    -------
    float
        loss value for the model and loader.

    """
    # Move some things to the model's device
    device = model.getDevice()
    edge_index_list = [ei.to(device) for ei in edge_index_list]
    clusters_list = [cl.to(device) for cl in clusters_list]
    
    model.eval()
    n_samples = 0
    loss_total = 0
    with torch.no_grad():
        for batch in loader_va:
            #print("New Batch")
            # Move some things to the model's device
            x = batch.x.to(device)
            y = batch.y.to(device)
            b = batch.batch.to(device)
            ei_list, cl_list = fixEdgesAndClustersForBatch(edge_index_list, clusters_list, b)
            yhat = model(x, ei_list, cl_list, b)
            loss = loss_function(yhat, y)
            n_samples += (b[-1] + 1)
            loss_total += (loss * (b[-1] + 1))
    return (loss_total / n_samples).cpu()

def fixEdgesAndClustersForBatch(edge_index_list_in, clusters_list_in, batch):
    """
    The edge indicies and clusters are fixed for the batch setting

    Parameters
    ----------
    edge_index_list_in : list of torch.Tensor (2, n_edges)
        A list of tensors containing the edges.
    clusters_list_in : list of torch.Tensor (2, n_nodes)
        A list of tensors with cluster assignments.
    batch : torch.Tensor (1, n_nodes * n_graphs)
        A tensor indicating which graph each node in a batch is part of

    Returns
    -------
    edge_index_list : list of torch.Tensor (2, n_edges * n_graphs)
        A list of tensors containing the edges.
    clusters_list : list of torch.Tensor (2, n_nodes * n_graphs)
        A list of tensors with cluster assignments.
    """
    # Make copies of the input lists because changes are made in-place
    edge_index_list = edge_index_list_in.copy()
    clusters_list = clusters_list_in.copy()
    
    # How many separate graphs are in the batch (this is the batch_size)
    n_graphs = batch[-1] + 1
    
    # Fix the first edge_index tensor
    edges = edge_index_list[0]
    n_nodes = torch.sum(batch == 0)
    for j in range(1, n_graphs):
        edge_index_list[0] = torch.cat((edge_index_list[0], edges + j * n_nodes), dim=1)
    
    # Fix the remaining edge_indicies and the clusters
    for i in range(len(clusters_list)):
        edges = edge_index_list[i+1]
        clusters = clusters_list[i]
        n_clusters = torch.max(clusters) + 1
        for j in range(1, n_graphs):
            clusters_list[i] = torch.cat((clusters_list[i], clusters + (j * n_clusters)), dim=0)
            edge_index_list[i+1] = torch.cat((edge_index_list[i+1], edges + j * n_clusters), dim=1)
    
    return edge_index_list, clusters_list

def trainEpochCNN(model, loader_tr, optimizer, loss_function):
    """
    Execute one epoch of training the model
    
    Parameters
    ----------
    model : ConvUNet
        The model to fit the parameters of.
    loader_tr : torch.utils.data.DataLoader
        A loader to use in training or validation.
    optimizer : torch.optim.Optimizer
        Parameter optimization routine.
    loss_function : func
        Function for computing the loss.

    Returns
    -------
    float
        loss value for the model and loader.

    """
    # Get model's device
    device = model.getDevice()
    
    model.train()
    n_samples = 0
    loss_total = 0
    for x, y in loader_tr:
        # Move some things to the model's device
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        yhat = model(x)
        loss = loss_function(yhat, y)
        loss.backward()
        optimizer.step()
        n_samples += x.size(0)
        loss_total += (loss * x.size(0))
    return (loss_total / n_samples).cpu().detach()

def validEpochCNN(model, loader_va, loss_function):
    """
    Execute one epoch of validating the model
    
    Parameters
    ----------
    model : ConvUNet
        The model to fit the parameters of.
    loader_tr : torch.utils.data.DataLoader
        A loader to use in training or validation.
    loss_function : func
        Function for computing the loss.

    Returns
    -------
    float
        loss value for the model and loader.

    """
    # Get model's device
    device = model.getDevice()
    
    model.eval()
    n_samples = 0
    loss_total = 0
    with torch.no_grad():
        for x, y in loader_va:
            # Move some things to the model's device
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            loss = loss_function(yhat, y)
            n_samples += x.size(0)
            loss_total += (loss * x.size(0))
    return (loss_total / n_samples).cpu().detach()

def predictModel(model, data_pr, params_pr):
    """
    Use the model to make predictions

    Parameters
    ----------
    model : torch.nn.Module (typically GraphUNet or ConvUNet)
        The model to apply to the data.
    data_pr : tuple of (x, {y}, {edge_index_list}, {clusters_list})
        Data to make predictions on using the model.
    params_pr : dict
        Dictionary of prediction parameters such as device.
        -> device : torch.Device
        -> loss_function : func
        -> print_freq : int

    Returns
    -------
    predict_output : dict
        Dictionary containing output from the prediction.

    """
    # Check that the model.type is "gnn" or "cnn"
    if not (model.type=="gnn" or model.type=="cnn"):
        raise Exception("Need model.type to be one of { gnn | cnn }")
        
    # Move things to the correct device
    model.to(params_pr["device"])
    if model.type == "gnn":
        edge_index_list = [ei.to(params_pr["device"]) for ei in data_pr[2]]
        clusters_list = [cl.to(params_pr["device"]) for cl in data_pr[3]]
    
    # Print a statement
    if params_pr["print_freq"] > 0:
        print("Using the model to predict on {} samples".format(data_pr[0].size(0)))
    
    # Initialize output
    yhat_size = list(data_pr[0].size())
    if model.type == "gnn":
        yhat_size[2] = model.channels_out
    elif model.type == "cnn":
        yhat_size[1] = model.channels_out
    yhat = torch.zeros(yhat_size, device=params_pr["device"], dtype=torch.double)
    
    # Loop passing samples through the model without doing gradient calcs
    model.eval()
    with torch.no_grad():
        t_start = time()
        for i in range(data_pr[0].size(0)):
            x = data_pr[0][i:i+1, :].to(params_pr["device"])
            if model.type == "gnn":
                b = torch.zeros((x.numel()), device=x.device)
                yhat[i, :, :] = model(x[0,:,:], edge_index_list, clusters_list, b)
            elif model.type == "cnn":
                yhat[i, :] = model(x)
            
    # Print a statement
    t = time() - t_start
    if params_pr["print_freq"] > 0:
        h = int(t // 3600)
        m = int((t - h*3600) // 60)
        s = (t - h*3600 - m*60)
        print("Prediction took {:02d}:{:02d}:{:02.0f} ({:5.3f} sec/sample)".format(h, m, s, t/data_pr[0].size(0)))
    
    # Move the model to cpu
    model.cpu()
    
    # Start the output
    predict_output = {
        "predict_time" : t,
        "yhat" : yhat
    }
    
    # Also compute the loss if data_pr[1] is provided
    if data_pr[1] is not None:
        loss = params_pr["loss_function"](yhat, data_pr[1])
        predict_output["loss"] = loss.cpu()
        
    return predict_output

def fitGCNM(model, data_tr, data_va, params_tr):
    """
    Fit a series of networks to the data using the GCNM.

    Parameters
    ----------
    model : torch.nn.Module (typically GraphUNet or ConvUNet)
        The model to fit the parameters of.
    data_tr : tuple of (x, y, {edge_index_list}, {clusters_list})
        Data to use for training the model.
    data_va : tuple of (x, y, {edge_index_list}, {clusters_list})
        Data to use for validating the model.
    params_tr : dict
        Dictionary of training parameters such as batch size.
        -> n_iterations : int
        -> batch_size : int
        -> device : torch.Device
        -> learning_rate : float
        -> max_epochs : int
        -> loss_function : func
        -> update_function : func
        -> patience : int
        -> print_freq : int

    Returns
    -------
    training_outputs : list of dict
        Dictionaries containing output from training each model.

    """
    # Check that the model.type is "gnn" or "cnn"
    if not (model.type=="gnn" or model.type=="cnn"):
        raise Exception("Need model.type to be one of { gnn | cnn }")
        
    # Initialize output as an empty list
    training_outputs = []
    
    # Loop through iterations, fitting the model and computing updates
    for i in range(params_tr["n_iterations"]):
        
        # Print a statement
        print()
        print("Working on iteration {} of {}".format(i+1, params_tr["n_iterations"]))
        
        # Fit the model
        model.resetParameters()
        training_output = model.fit(data_tr, data_va, params_tr)
        training_output["state_dict"] = deepcopy(model.state_dict())
        
        # Append training output
        training_outputs.append(training_output)
        
        # If on the last iteration, just stop here
        if i == params_tr["n_iterations"] - 1:
            break
        
        # Make predictions using the model on both the tr and va data
        yhats_tr = model.predict(data_tr, params_tr)["yhat"]
        yhats_va = model.predict(data_va, params_tr)["yhat"]
        
        # Compute updates using the current iterate and model outputs
        # Note that the training and validation are combined and then separated
        xs = torch.cat((data_tr[0], data_va[0]), dim=0)
        yhats = torch.cat((yhats_tr, yhats_va), dim=0)
        xs = params_tr["update_function"](xs, yhats)
        x_tr = xs[0:data_tr[0].size(0), :]
        x_va = xs[data_tr[0].size(0): , :]
        
        # Update the training and validation data for the next iterations
        data_tr = (x_tr, *data_tr[1:])
        data_va = (x_va, *data_va[1:])
        
    return training_outputs

def predictGCNM(model, state_dicts, data_pr, params_pr):
    """
    Make predictions on the data using a series of trained models

    Parameters
    ----------
    model : torch.nn.Module (typically GraphUNet or ConvUNet)
        The model to fit the parameters of.
    state_dicts : list of dicts
        List of state dictionaries that can be loaded into the model.
    data_pr : tuple of (x, y, edge_index_list, clusters_list)
        Data to make predictions on using the models.
    params_pr : dict
        Dictionary of prediction parameters such as device.
        -> device : torch.Device
        -> loss_function : func
        -> update_function : func

    Returns
    -------
    predict_outputa : list of dicts
        Dictionary containing output from the predictions.

    """
    # Initialize output as an empty list
    predict_outputs = []
    
    # Loop through iterations, loading the state dicts and such
    for i, state_dict in enumerate(state_dicts):
        
        # Print a statement
        print()
        print("Working on iteration {} of {}".format(i+1, len(state_dicts)))
        
        # Load the state dict into the model
        model.load_state_dict(state_dict)
        
        # Make predictions using the model
        predict_output = model.predict(data_pr, params_pr)
        
        # Append training output
        predict_outputs.append(predict_output)
        
        # If on the last iteration, just stop here
        if i == len(state_dicts) - 1:
            break
        
        # Compute updates using the current iterate and model outputs
        yhats_pr = predict_output["yhat"]
        x_pr = params_pr["update_function"](data_pr[0], yhats_pr)
        
        # Update the training and validation data for the next iterations
        data_pr = (x_pr, *data_pr[1:])
        
    return predict_outputs


