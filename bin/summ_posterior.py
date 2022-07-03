import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import common
import result_serializer
# import clustermaker
# import inputparser

import json
import util
import plotutil
# import diversity_indices as di

# import plotly.graph_objs as go
# import plotly.io as pio

def write_header(runid, outf):
  if runid is not None:
    title = '%s posterior summary' % runid
  else:
    title = 'Posterior summary'

  # TODO: try the ELK layout algorithm instead of Klay. Currently,
  # cytoscape.js-elk is a little outdated, and seems primed for an upgrade via
  # this PR: https://github.com/cytoscape/cytoscape.js-elk/pull/15. It has not
  # been merged as of March 2020, though, so I will wait until it is.
  print('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>%s</title>' % title, file=outf)
  for jsurl in (
    'https://d3js.org/d3.v5.min.js',
    #'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.14.0/cytoscape.min.js',

    'https://cdn.jsdelivr.net/npm/klayjs@0.4.1/klay.min.js',
    'https://cdn.jsdelivr.net/npm/cytoscape-klay@3.1.3/cytoscape-klay.min.js',

    #'https://cdn.jsdelivr.net/npm/cytoscape-euler@1.2.2/cytoscape-euler.min.js',

    #'https://cdn.jsdelivr.net/npm/weaverjs@1.2.0/dist/weaver.min.js',
    #'https://cdn.jsdelivr.net/npm/cytoscape-spread@3.0.0/cytoscape-spread.min.js',

    'https://cdn.jsdelivr.net/npm/numeric@1.2.6/numeric-1.2.6.min.js',
    'https://cdn.jsdelivr.net/npm/layout-base@1.0.2/layout-base.min.js',
    'https://cdn.jsdelivr.net/npm/cose-base@1.0.1/cose-base.min.js',
    'https://cdn.jsdelivr.net/npm/cytoscape-fcose@1.2.0/cytoscape-fcose.min.js',

    # We need these just to display edge weight over edges on hover. Bleh.
    'https://unpkg.com/popper.js@1',
    'https://unpkg.com/tippy.js@5',
    'https://cdn.jsdelivr.net/npm/cytoscape-popper@1.0.6/cytoscape-popper.min.js',

    'https://cdn.jsdelivr.net/npm/canvas2svg@1.0.16/canvas2svg.min.js',
    'https://cdn.jsdelivr.net/npm/cytoscape-svg@0.2.0/cytoscape-svg.min.js',
    'https://cdn.jsdelivr.net/npm/file-saver@2.0.2/dist/FileSaver.min.js',
  ):
    print('<script type="text/javascript" src="%s"></script>' % jsurl, file=outf)
  for jsfn in ('tree_plotter.js', 'posterior_summ.js', 'util.js'):
    print('<script type="text/javascript">%s</script>' % plotutil.read_file(jsfn), file=outf)
  print('<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">', file=outf)
  print('<style type="text/css">%s</style>' % plotutil.read_file('tree.css'), file=outf)
  print('</head><body><main role="main" class="container">', file=outf)
  if runid is not None:
    print('<h1>%s</h1>' % runid, file=outf)

#NOTE: took the below from pairtree's util file. I striped out whatever wasn't needed for sc_pairtree
def make_tree_struct(struct, count, llh, prob, phi, clusters, sampnames):
  #K, S = phi.shape
  #eta = calc_eta(struct, phi)

  tree = {
    'phi': -1, #phi.tolist(),
    'eta': -1, #eta.tolist(),
    'llh': float(llh),
    'nlglh': -1,#float(calc_nlglh(llh, K, S)),
    'prob': float(prob),
    'count': int(count),
    'parents': struct.tolist(),
    'samples': ['no sample names in sc_pairtree'],#sampnames,
  }
  return tree

# def summarize(results, params, supervars, outf):
def summarize(results, outf):
  prob = results.get('prob')
  N = len(prob)
  assert np.array_equal(np.argsort(-prob, kind='stable'), np.arange(N))

  _make_struct = lambda tidx: make_tree_struct(
    results.get('struct')[tidx],
    results.get('count')[tidx],
    results.get('llh')[tidx],
    prob[tidx],
    -1, #results.get('phi')[tidx],
    -1, #results.get('clusters'),
    -1, #results.get('sampnames'),
  )

  limit = 20
  structs = [_make_struct(tidx) for tidx in range(min(N, limit))]
  json_trees = {
    'structs': structs,
    # 'samp_colours': params.get('samp_colours', None),
    'samp_colours': None,
  }

  print("<script type=\"text/javascript\">var tree_json = '%s'; var results = JSON.parse(tree_json);</script>" % json.dumps(json_trees), file=outf)
  print('<h2>Trees</h2>', file=outf)
  print('<ul><li>Number of unique trees: %s</li></ul>' % N, file=outf)
  print('<table id="trees" class="table table-striped"><thead><tr><th>Index</th><th>Posterior</th><th>nLgLh</th><th>Count</th><th>Structure</th></tr></thead><tbody class="container"></tbody></table>', file=outf)
  print(plotutil.js_on_load("(new PosteriorSumm()).plot(results, '#trees .container');"), file=outf)

def _make_congraph(results):
  adjms = np.array([util.convert_parents_to_adjmatrix(struct) for struct in results.get('struct')])
  weights = results.get('prob')
  assert len(weights) == len(adjms)
  assert np.isclose(1, np.sum(weights))
  graph = np.sum(weights[:,np.newaxis,np.newaxis] * adjms, axis=0)
  np.fill_diagonal(graph, 0)

  assert np.allclose(1, graph[graph > 1])
  graph[graph > 1] = 1

  parent_sum = np.sum(graph, axis=0)
  assert parent_sum[0] == 0 and np.allclose(1, parent_sum[1:])
  assert np.all(0 <= graph) and np.all(graph <= 1)

  return graph

# def _calc_di(results, samps=1000):
#   structs = results.get('struct')
#   phis = results.get('phi')
#   probs = results.get('prob')
#   clusters = results.get('clusters')
#   assert np.isclose(1, np.sum(probs))

#   cdi = []
#   cmdi = []
#   cadi = []
#   for idx in range(samps):
#     tidx = util.sample_multinom(probs)
#     eta = util.calc_eta(structs[tidx], phis[tidx])
#     cdi.append(di.calc_cdi(eta))
#     cmdi.append(di.calc_cmdi(eta, clusters, structs[tidx]))
#     cadi.append(di.calc_cadi(eta, structs[tidx]))

  # Size of each array: PxS, given P posterior samples and S tissue samples
  return (np.array(cdi), np.array(cmdi), np.array(cadi))

def _test_didxs(didxs, didx_type, sidx1, sidx2, alt, sampnames):
  import scipy.stats
  stat, pval = scipy.stats.wilcoxon(didxs[didx_type][:,sidx1], didxs[didx_type][:,sidx2], alternative=alt)
  print(didx_type, sampnames[sidx1], sampnames[sidx2], alt, pval)

# def _plot_di(results, visible_sampidxs, outf):
#   didxs = {K: I for K, I in zip(('cdi', 'cmdi', 'cadi'), _calc_di(results))}
#   sampnames = results.get('sampnames')
#   print('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>', file=outf)
#   print('<h2>Diversity indices</h2>', file=outf)

#   titles = {
#     'cdi': 'Clone diversity index',
#     'cmdi': 'Clone and mutation diversity index',
#     'cadi': 'Clone and ancestor diversity index',
#   }

#   for K in didxs.keys():
#     traces = [dict(
#       type = 'box',
#       name = sampname,
#       y = didxs[K][:,sidx],
#       showlegend = False,
#       boxpoints = 'all',
#       boxmean = True,
#       pointpos = 1.5,
#     ) for sidx, sampname in enumerate(sampnames) if visible_sampidxs is None or sidx in visible_sampidxs]
#     layout = go.Layout(
#       template = 'seaborn',
#       title = '{di_name}'.format(di_name = titles[K]),
#       xaxis = {'title': 'Sample'},
#       yaxis = {'title': '{di_name} (bits)'.format(di_name = K.upper())},
#     )
#     fig = go.Figure(data=traces, layout=layout)

#     # Example hack showing how you can do Wilcoxon signed-rank test on
#     # diversity indices for two particular samples. Eventually this
#     # functionality should be broken out into its own discrete script.
#     #_test_didxs(didxs, K, 0, 46, 'less', sampnames)
#     #_test_didxs(didxs, K, 70, 72, 'greater', sampnames)

#     html = pio.to_html(
#       fig,
#       include_plotlyjs=False,
#       full_html=False,
#       config = {
#         'showLink': True,
#         'toImageButtonOptions': {
#           'format': 'svg',
#           'width': 750,
#           'height': 450,
#         },
#       }
#     )
#     print(html, file=outf)

def _plot_congraph(congraph, outf):
  print('''
  <h2>Consensus graph</h2>
  <h5>Minimum spanning-tree threshold: <span id="spanning_threshold"></span></h5>
  <form>
    <div class="row">
      <div class="col">
        <label for="threshold_chooser">Edge threshold: <span id="congraph_threshold"></span></label>
        <input type="range" id="threshold_chooser" class="form-control-range">
      </div>

      <div class="col">
        <label for="layout_chooser">Layout</label>
        <select class="form-control" id="layout_chooser">
        </select>
      </div>

      <div class="col">
        <label>Export graph</label>
        <div id="exporters">
          <button id="export_svg" type="button" class="btn btn-primary">Export SVG</button>
          <button id="export_png" type="button" class="btn btn-primary">Export PNG</button>
        </div>
      </div>
    </div>
  </form>

  <style type="text/css">
  #congraph { width: 100%%; height: 400px; }
  </style>
  <div id="congraph" class="row"></div>

  <script type="text/javascript">
  var congraph_json = '%s';
  var congraph = JSON.parse(congraph_json);
  </script>
  ''' % json.dumps(congraph.tolist()), file=outf)
  print(plotutil.js_on_load("(new CongraphPlotter()).plot(congraph, '#congraph', '#congraph_threshold');"), file=outf)

def write_footer(outf):
  print('</main></body></html>', file=outf)

def _choose_plots(to_plot, to_omit, all_choices):
  # Duplicate set.
  if to_plot is None:
    plot_choices = set(all_choices)
  else:
    plot_choices = set(to_plot)

  if to_omit is not None:
    plot_choices -= to_omit

  assert plot_choices.issubset(all_choices)
  return plot_choices

def main():
  all_plot_choices = (
    'posterior_summ',
    'congraph',
    # 'diversity_indices', # JB: removing for now, since I don't even do any clustering at the moment
  )

  parser = argparse.ArgumentParser(
    description='Summarize the posterior distribution over clone trees sampled by Pairtree',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--runid')
  parser.add_argument('--plot', dest='plot_choices', type=lambda s: set(s.split(',')),
    help='Things to plot; by default, plot everything')
  parser.add_argument('--omit-plots', dest='omit_plots', type=lambda s: set(s.split(',')),
    help='Things to omit from plotting; overrides --plot')
  # parser.add_argument('ssm_fn')
  # parser.add_argument('params_fn')
  parser.add_argument('results_fn')
  parser.add_argument('html_out_fn')
  args = parser.parse_args()

  np.seterr(divide='raise', invalid='raise', over='raise')

  plot_choices = _choose_plots(args.plot_choices, args.omit_plots, all_plot_choices)

  results = result_serializer.Results(args.results_fn)
  # variants = inputparser.load_ssms(args.ssm_fn)
  # params = inputparser.load_params(args.params_fn)
  # supervars = clustermaker.make_cluster_supervars(results.get('clusters'), variants)
  # supervars = [supervars[vid] for vid in common.sort_vids(supervars.keys())]

  # visible_sampidxs = plotutil.hide_samples(params['samples'], params.get('hidden_samples'))

  with open(args.html_out_fn, 'w') as outf:
    write_header(args.runid, outf)

    if 'congraph' in plot_choices:
      congraph = _make_congraph(results)
      _plot_congraph(congraph, outf)

    if 'posterior_summ' in plot_choices:
      # summarize(results, params, supervars, outf)
      summarize(results, outf)

    # if 'diversity_indices' in plot_choices:
    #   _plot_di(results, visible_sampidxs, outf)

    write_footer(outf)

if __name__ == '__main__':
  main()
