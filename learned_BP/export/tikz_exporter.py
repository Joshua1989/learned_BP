#########################################################################################################
# Tikz figure generating
#########################################################################################################
class TikzExporter:
    def __init__(self):
        self.tikz_template = '''
\\vspace*{{-7mm}}
\\hspace*{{-6mm}}
\\begin{{tikzpicture}}

\\begin{{semilogyaxis}}[%
width=3.9in,
height=2.55in,
scale only axis,
every axis plot/.append style={{very thick}},
every outer x axis line/.append style={{thick,darkgray!60!black}},
every x tick label/.append style={{font=\\color{{darkgray!60!black}}}},
xmin={xmin}, xmax={xmax},
xlabel={{{xlabel}}},
xmajorgrids,
every outer y axis line/.append style={{darkgray!60!black}},
every y tick label/.append style={{font=\\color{{darkgray!60!black}}}},
ymin={ymin}, ymax={ymax},
yminorticks=true,
ylabel={{{ylabel}}},
ymajorgrids,
yminorgrids,
title={{{title}}},
grid style={{solid}},
legend style={{draw=darkgray!60!black,fill=white,legend cell align=left,font=\\footnotesize, legend pos=south west}}]

{curves}
        
\\end{{semilogyaxis}}
\\end{{tikzpicture}}
'''

        self.curve_template = '''
\\addplot [
    thick,
    color={color},
    {linestyle},
    mark={marker},
    mark options={{solid}}
]
coordinates{{
 {coord}
}};
\\addlegendentry{{{legend}}}
'''
        self.plot_args = {
            'xmin': 0,
            'xmax': 6,
            'ymin': 0.0000001,
            'ymax': 1,
            'xlabel': '',
            'ylabel': '',
            'title': '',
            'curves': ''
        }

    def set_xlim(self, xlim):
        self.plot_args['xmin'] = xlim[0]
        self.plot_args['xmax'] = xlim[1]

    def set_ylim(self, ylim):
        self.plot_args['ymin'] = ylim[0]
        self.plot_args['ymax'] = ylim[1]

    def set_xlabel(self, xlab):
        self.plot_args['xlabel'] = xlab

    def set_ylabel(self, ylab):
        self.plot_args['ylabel'] = ylab

    def set_title(self, title):
        self.plot_args['title'] = title

    def add_plots(self, results):
        self.plot_args['curves'] = ''.join([self.curve_template.format(**self.format(k, v)) for k,v in sorted(results.items())])

    def export(self, file_name):
        with open(file_name, 'w') as fp:
            fp.write(self.tikz_template.format(**self.plot_args))

    def format(self, name, coord):
        curve_args = {
            'legend': name,
            'color': 'black',
            'marker': 'o',
            'linestyle': 'solid',
            'coord': coord
        }

        # Customize the style below
        if 'cyc' not in name and 'oc' not in name:
            # For curse use standard PC matrix, dash dot line style, + marker
            curve_args['linestyle'] = 'dash dot'
            curve_args['marker'] = '+'
        else:
            # For curse use standard PC matrix, solid line style, o marker
            curve_args['linestyle'] = 'solid'
            curve_args['marker'] = 'o'

        curve_args['legend'] = ''
        if 'cyc' in name:     
            curve_args['legend'] += 'circ '
        if 'oc' in name:        
            curve_args['legend'] += 'All MWPC '
        if 'standard' in name:  
            curve_args['legend'] += 'plain BP'
            curve_args['color'] = 'red!70!black'
        if 'simple' in name:    
            curve_args['legend'] += 'simple'
            curve_args['color'] = 'green!70!black'
        if 'RNN' in name:       
            curve_args['legend'] += 'RNN'
            curve_args['color'] = 'blue!70!black'
        if 'FF' in name:        
            curve_args['legend'] += 'FF'
            curve_args['color'] = 'magenta!70!black'

        return curve_args