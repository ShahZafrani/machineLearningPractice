class feed_forward_neural_network (object):

    # create constructor
    def __init__(self, size_list):
        self.num_layers = len(size_list)
        self.layer_sizes = size_list
        
