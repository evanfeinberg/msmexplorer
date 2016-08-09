import numpy as np
import networkx as nx
import seaborn.apionly as sns
from matplotlib import pyplot as pp

from ..palettes import msme_rgb

__all__ = ['plot_pop_resids', 'plot_msm_network', 'plot_timescales']


def plot_pop_resids(msm, **kwargs):
    if hasattr(msm, 'all_populations_'):
        msm_pop = msm.populations_.mean(0)
    elif hasattr(msm, 'populations_'):
        msm_pop = msm.populations_

    raw_pop = msm.countsmat_.sum(1) / msm.countsmat_.sum()
    ax = sns.jointplot(np.log10(raw_pop), np.log10(msm_pop), kind='resid',
                       **kwargs)
    ax.ax_joint.set_xlabel('Raw Populations', size=20)
    ax.ax_joint.set_ylabel('Residuals', size=20)

    return ax


def plot_msm_network(msm, pos=None, node_color='c', node_size=300,
                     edge_color='k', ax=None, with_labels=True, **kwargs):
    if hasattr(msm, 'all_populations_'):
        tmat = msm.all_transmats_.mean(0)
    elif hasattr(msm, 'populations_'):
        tmat = msm.transmat_

    graph = nx.Graph(tmat)

    if not ax:
        ax = pp.gca()

    nx.draw_networkx(graph, pos=pos, node_color=node_color,
                     edge_color=edge_color, ax=ax, **kwargs)

    return ax


def plot_timescales(msm, n_timescales=None, error=None, sigma=2, colors=None,
                    xlabel=None, ylabel=None, ax=None):

    if hasattr(msm, 'all_timescales_'):
        timescales = msm.all_timescales_.mean(0)
        if not error:
            error = (msm.all_timescales_.std(0) /
                     msm.all_timescales_.shape[0] ** 0.5)
    elif hasattr(msm, 'timescales_'):
        timescales = msm.timescales_
        if not error:
            error = np.nan_to_num(msm.uncertainty_timescales())

    if n_timescales:
        timescales = timescales[:n_timescales]
        error = error[:n_timescales]
    else:
        n_timescales = timescales.shape[0]

    ymin = 10 ** np.floor(np.log10(np.nanmin(timescales)))
    ymax = 10 ** np.ceil(np.log10(np.nanmax(timescales)))

    if not ax:
        ax = pp.gca()
    if not colors:
        colors = list(msme_rgb.values())

    for i, item in enumerate(zip(timescales, error)):
        t, s = item
        color = colors[i % len(colors)]
        ax.errorbar([0, 1], [t, t], c=color)
        if s:
            for j in range(1, sigma + 1):
                ax.fill_between([0, 1], y1=[t - j * s, t - j * s],
                                y2=[t + j * s, t + j * s],
                                color=color, alpha=0.2 / j)

    ax.xaxis.set_ticks([])
    if xlabel:
        ax.xaxis.set_label_text(xlabel, size=18, labelpad=18)
    if ylabel:
        ax.yaxis.set_label_text(ylabel, size=18)
    ax.set_yscale('log')
    ax.set_ylim([ymin, ymax])

    autoAxis = ax.axis()
    rec = pp.Rectangle((autoAxis[0], 100),
                       (autoAxis[1] - autoAxis[0]),
                       ymax, fill=False, lw=2)
    rec = ax.add_patch(rec)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    return ax

def construct_graph(msm_modeler_dir, clusterer_dir, n_clusters, tica_lag_time=5, msm_lag_time=10, graph_file="~/graph_file.graphml", msm_object=None, clusterer_object=None,
									  inactive = None, active = None, pnas_clusters_averages = None, 
									  tica_clusters_averages = None, docking=None, macrostate = None, 
									  cluster_attributes=None, msm_attributes=None):
	
    """
    Construct a .graphml graph based on an MSM and attributes of clusters and/or MSM states.
    Saves .graphml graph to disk and returns it as well. 

    *needs networkx python package to use*
    
    Parameters
    ----------
    msm_modeler_dir: location on disk of verboseload loadable msm object 
    clusterer_dir: location on disk of verboseload loadable clusterer object 
    n_clusters: number of clusters
    tica_lag_time: tica lag time
    msm_lag_time: msm lag time 
    graph_file: location on disk for saving graphml file 
    msm_object: pass msm object directly instead of loading from disk 
    clusterer_object: pass clusterer object directly instead of loading from disk 
    cluster_attributes: dictionary that maps names of attributes to lists of size n_clusters
    	where each entry in the list is the value of that attribute for that cluster. for example,
    	if n_clusters=3, an example cluster_attributes dict might be: 
    		cluster_attributes = {'tyr75-his319_dist': [7.0, 6.0, 8.0], 'phe289-chi2': [90.0, 93.0, 123.2]}
    msm_attributes: dictionary that maps names of attributes to lists of size n_msm_states
    	where each entry in the list is the value of that attribute for that msm state. for example,
    	if n_msm_states=3, an example cluster_attributes dict might be: 
    		msm_attributes = {'tyr75-his319_dist': [7.0, 6.0, 8.0], 'phe289-chi2': [90.0, 93.0, 123.2]}
    """

	if clusterer_object is None:
		clusterer = verboseload(clusterer_dir)
	else:
		clusterer = clusterer_object
	n_clusters = np.shape(clusterer.cluster_centers_)[0]

	labels = clusterer.labels_

	if not os.path.exists(msm_modeler_dir):
		if msm_object is not None:
			msm_modeler = msm_object
		else:
			msm_modeler = MarkovStateModel(lag_time=msm_lag_time, n_timescales = 5, sliding_window = True, verbose = True)
		print(("fitting msm to trajectories with %d clusters and lag_time %d" %(n_clusters, msm_lag_time)))
		msm_modeler.fit_transform(labels)
		verbosedump(msm_modeler, msm_modeler_dir)
	else:
		msm_modeler = verboseload(msm_modeler_dir)
	graph = nx.DiGraph()
	mapping = msm_modeler.mapping_
	inv_mapping = {v: k for k, v in list(mapping.items())}
	transmat = msm_modeler.transmat_

	for i in range(0, msm_modeler.n_states_):
		for j in range(0, msm_modeler.n_states_):
			prob = transmat[i][j]
			if prob > 0.0:
				if prob < 0.001: prob = 0.001
				original_i = inv_mapping[i]
				original_j = inv_mapping[j]
				graph.add_edge(original_i, original_j, prob = float(prob), inverse_prob = 1.0 / float(prob))

	print("Number of nodes in graph:")
	print((graph.number_of_nodes()))

	if inactive is not None:
		scores = convert_csv_to_map_nocombine(inactive)
		for cluster in list(scores.keys()):
			cluster_id = int(cluster[7:len(cluster)])
			if cluster_id in graph.nodes():
				score = scores[cluster][0]
				graph.node[cluster_id]["inactive_pnas"] = score

	if active is not None:
		scores = convert_csv_to_map_nocombine(active)
		for cluster in list(scores.keys()):
			cluster_id = int(re.search(r'\d+',cluster).group()) 
			if cluster_id in graph.nodes():
				score = scores[cluster][0]
				graph.node[cluster_id]["active_pnas"] = score

	if pnas_clusters_averages is not None:
		scores = convert_csv_to_map_nocombine(pnas_clusters_averages)
		for cluster in list(scores.keys()):
			cluster_id = int(re.search(r'\d+',cluster).group()) 
			if cluster_id in graph.nodes():
				graph.node[cluster_id]["tm6_tm3_dist"] = scores[cluster][0]
				graph.node[cluster_id]["rmsd_npxxy_active"] = scores[cluster][2]
				graph.node[cluster_id]["rmsd_connector_active"] = scores[cluster][4]

	if tica_clusters_averages is not None:
		scores = convert_csv_to_map_nocombine(tica_clusters_averages)
		for cluster in list(scores.keys()):
			cluster_id = int(re.search(r'\d+',cluster).group()) 
			if cluster_id in graph.nodes():
				for i in range(0,len(scores[cluster])):
					graph.node[cluster_id]["tIC%d" %(i+1)] = scores[cluster][i]

	if docking is not None:
		scores = convert_csv_to_map_nocombine(docking)
		for cluster in list(scores.keys()):
			cluster_id = int(cluster[7:len(cluster)])
			if cluster_id in graph.nodes():
				score = scores[cluster][0]
				graph.node[cluster_id]["docking"] = score

	if macrostate is not None:
		macromodel = verboseload(macrostate)
		for cluster_id in range(0, n_clusters):
			if cluster_id in graph.nodes():
				microstate_cluster_id = mapping[cluster_id]
				macrostate_cluster_id = macromodel.microstate_mapping_[microstate_cluster_id]
				#print(macrostate_cluster_id)
				graph.node[cluster_id]["macrostate"] = int(macrostate_cluster_id)

	if cluster_attributes is not None:
		for attribute in cluster_attributes.keys():
			for cluster_id in mapping.keys():
				graph.node[cluster_id][attribute] = float(cluster_attributes[attribute][cluster_id])


	if msm_attributes is not None:
		for attribute in msm_attributes.keys():
			for cluster_id in mapping.keys():
				graph.node[cluster_id][attribute] = float(msm_attributes[attribute][mapping[cluster_id]])

	nx.write_graphml(graph, graph_file)
	return(graph)
