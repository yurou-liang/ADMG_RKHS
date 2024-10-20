import os
import shutil
import pickle
import itertools
from ananke.graphs import ADMG


def pprint_pag(G):
    """
    Function to pretty print out PAG edges

    :param G: Ananke ADMG with 'pag_edges' attribute.
    :return: None.
    """
    print ('-'*10)
    print (f'Nodes: {list(G.vertices.keys())}')
    for edge in G.pag_edges:
        print (f'{edge["u"]} {edge["type"]} {edge["v"]}')


def write_admg_to_file(G, filename):
    """
    Function to write ADMG to file in correct format for PAG conversion.

    :param G: Ananke ADMG.
    :return: None.
    """

    f = open(filename, 'w')
    f.write("Graph Nodes:\n")
    f.write(','.join(['X'+str(v) for v in G.vertices]) + '\n\n')
    f.write("Graph Edges:\n")
    counter = 1
    for Vi, Vj in G.di_edges:
        f.write(str(counter) + '. X' + str(Vi) + ' --> X' + str(Vj) + '\n')
        counter += 1
    for Vi, Vj in G.bi_edges:
        f.write(str(counter) + '. X' + str(Vi) + ' <-> X' + str(Vj) + '\n')
        counter += 1
    f.close()


def inducing_path(G, Vi, Vj):
    """
    Checks if there is an inducing path between Vi and Vj in G.

    :return: boolean indicator whether there is an inducing path.
    """

    # easy special case of directed adjacency
    if Vi in G.parents([Vj]) or Vj in G.parents([Vi]) or Vi in G.siblings([Vj]):
        return True

    ancestors_ViVj = G.ancestors([Vi, Vj])
    visit_stack = [s for s in G.siblings([Vi]) if s in ancestors_ViVj]
    visit_stack += [c for c in G.children([Vi]) if c in ancestors_ViVj]
    visited = set()

    while visit_stack:

        if Vj in visit_stack or Vj in G.parents(visit_stack):
            return True

        v = visit_stack.pop()
        visited.add(v)
        visit_stack.extend(set([s for s in G.siblings([v]) if s in ancestors_ViVj]) - visited)
    return False


def mag_projection(G):
    """
    Get MAG projection of an ADMG G.

    :param G: Ananke ADMG.
    :return: Ananke ADMG corresponding to a MAG.
    """

    G_mag = ADMG(G.vertices)

    # iterate over all vertex pairs
    for Vi, Vj in itertools.combinations(G.vertices, 2):

        # check if there is an inducing path
        if inducing_path(G, Vi, Vj):
            # connect based on ancestrality
            if Vi in G.ancestors([Vj]):
                G_mag.add_diedge(Vi, Vj)
            elif Vj in G.ancestors([Vi]):
                G_mag.add_diedge(Vj, Vi)
            else:
                G_mag.add_biedge(Vi, Vj)

    return G_mag


def admg_to_pag(G, tmpdir='tmp/'):
    """
    Write an ADMG G to file, and then convert it to a PAG using tetrad

    :param G: Ananke ADMG.
    :return: Ananke ADMG with an 'pag_edges' attribute corresponding to a PAG.
    """

    os.makedirs(tmpdir, exist_ok=True)

    # write to disk for tetrad
    mag = mag_projection(G)
    write_admg_to_file(mag, f'{tmpdir}/G.mag')

    # convert to pag and write to disk
    os.system(f'java -classpath "utils/tetrad-lib-7.6.5.jar:utils/xom-1.3.5.jar:utils/" convertMag2Pag {tmpdir}/G.mag')

    # load back into new ADMG and return
    lines = open(f'{tmpdir}/G.mag.pag', 'r').read().strip().split('\n')
    nodes = lines[1].split(';')
    nodes = [str(node[1:]) for node in nodes] # remove X

    edges = []
    for line in lines[4:]:
      edge = line.split('. ')[1].split(' ')
      edges.append({'u':str(edge[0][1:]), 'v':str(edge[2][1:]), 'type':edge[1]})

    G = ADMG(nodes)
    G.pag_edges = edges

    # cleanup
    shutil.rmtree(tmpdir)

    return G
