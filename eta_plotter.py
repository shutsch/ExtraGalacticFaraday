import os
import nifty8 as ift
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

class Eta_Plotter():
    def __init__(self, args):
        self.emodel=args['emodel']
        self.ecomponents = args['ecomponents']
        self.params = args['params']
       


    def plot(self, name, path, string=None, **kwargs):


        latest_sample_list=ift.ResidualSampleList.load(f'{self.params["params_inference.results_path"]}pickle/last')

        plot_obj=[s for s in latest_sample_list.iterator()]
        if string is None:
           string = ''
        eta_path = path + 'eta/'+ '/'
        if not os.path.exists(eta_path):
          os.makedirs(eta_path)
        plot = ift.Plot()
        if isinstance(plot_obj, list):
           sc = ift.StatCalculator()
           for sample in plot_obj:
             sc.add(sample[name])
           m = sc.mean
           print(m.val)
        else:
          m =plot_obj
        plot.add(m, title=string, **kwargs)
        plot.output(name=eta_path + name +'_' + string + ".png")
        print('eta mean min', m.val.min())
        print('eta mean max', m.val.max())
        
        plt.clf()

