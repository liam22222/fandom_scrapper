from data_utils import get_redirects_mapper, create_reverse_mapper, getting_model
from network import get_mapper
from visualization import create_graph, save_graph_for_gephi, draw_graph

mapper, paragraph_mapper = get_mapper()
model = getting_model()

# redirects = get_redirects_mapper(mapper)
# reverse_mapper = create_reverse_mapper(mapper, redirects)
# reverse_mapper_sorted = dict(sorted(reverse_mapper.items(), key=lambda x: len(x[1]), reverse=True))
# print('Mapper:', mapper)
# graph, sizes = create_graph(reverse_mapper_sorted)
# draw_graph(graph, sizes)
