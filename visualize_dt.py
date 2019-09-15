import argparse
import pickle as pk
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus


def main(args):
    with open(args.input, 'rb') as f:
        tree = pk.load(f)
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data, filled=True)
    out_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    out_graph.write_svg(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    main(args)
