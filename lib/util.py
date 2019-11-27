class plotter:  
    """
    Graph plot generator
    
    Parameters
    ----------
    name : str
        Name of graph to plot.
    scale: str
        Scale of the y axis. Options supported are 'linear' and 'log'.    
    maxRange : list, optional
        Maximum range of x and y axis represented by a list [max_x, max_y]. By default [1,1].
    """
    def __init__(self, name, scale, maxRange=[1,1]):       
        self.name = name
        self.scale = scale
        self.maxRange = maxRange

    def draw(self, scale=None):
        """
        Draws and displays the graph to screen. Additionally, it adds the 
        graph :class:`handle` to a global Handle resource list.
        
        Parameters
        ----------
        scale : str, optional
            scale of the y axis, by default None
        
        Returns
        -------
        int
            Returns int 0 to indicate graph was succesfully plotted.

        Raises
        ------
        NotImplementedError
            If no appropriate scale type is set for graph axes.

        TODO
        ----
        Implement true graph plotting function.     
        """

        if scale is None and self.scale != 'linear' and self.scale != 'log':
            raise NotImplementedError("Scale for graph axes not set.")

        print("Succesfully generated {} plot on a {} scale.".format(self.name, self.scale))
        return 0
