import collections

from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
import streamlit as st

import rg_database_interactions as rgdb
import database_interactions as daint


def create_content_str(item: rgdb.BaseNode, recent_prices: collections.defaultdict[int: [int | float]],
                       spells: dict[int: (str, str, int, str, str)]):
    if isinstance(item, rgdb.ItemNode):
        price = recent_prices[item.id]
        content_str = f"""
Item {item.name}

Id: {item.id}
Amount: ({item.required_amount})

Price: {"?" if price == float("inf") else daint.price2gold(price*item.required_amount)}
"""
    else:
        _, name, cooldown, profession_name, skill = spells.get(item.id, ("", "", 0, "", 0))
        content_str = f"""
Spell {item.name}

Id: {item.id}

{profession_name} {">=" if profession_name and skill else ""} {"" if not skill else skill}

{"Cooldown: " if cooldown else ""}{cooldown if cooldown else ""}{"s" if cooldown else ""}.
"""
    return content_str


def create_flow(root: rgdb.ItemNode, recent_prices: collections.defaultdict[int: [int | float]],
                spells: dict[int: (str, str, int, str, str)]):
    # spells has the following format:
    # spell.name_en, spell.name_de, spell.cooldown, spell.profession_name, spell.skill

    # traverse the graph using layer wise bfs and create the nodes and edges
    flow_node = StreamlitFlowNode(id='0.0', pos=(0, 0),
                                  data={'content': create_content_str(root, recent_prices, spells)},
                                  node_type='input', source_position='right', target_position='left',
                                  style={'border': '5px solid black'})
    layer = [(root, flow_node)]
    nodes = [flow_node]
    edges = []
    level = 1
    graph_ids = {flow_node.id}
    while layer:

        # go through the stack which corresponds to the current layer
        children = []
        for node, flnode in layer:
            for child in node.children.values():

                # create a new node
                flow_node = StreamlitFlowNode(id=f'{level+1}.{len(children)}', pos=(level*300, 50*len(children)),
                                              data={'content': create_content_str(child, recent_prices, spells)},
                                              node_type='default', source_position='right', target_position='left',
                                              style={'border': '5px solid black'} if child.marked else dict())

                # create a new edge
                flow_edge = StreamlitFlowEdge(f'{flnode.id}-{flow_node.id}', str(flnode.id), str(flow_node.id),
                                              animated=True, marker_end={'type': 'arrow'})

                # append both to the nodes and edges
                edges.append(flow_edge)
                nodes.append(flow_node)

                # append the child to the new level
                children.append((child, flow_node))

                # put the graph elements into the dict
                graph_ids.add(flow_node.id)

        # update the level
        level += 1
        layer = children
    state = StreamlitFlowState(nodes, edges)
    return state, graph_ids


def delete_nodes(node_ids: set[str]):
    print([node.id for node in st.session_state.curr_state.nodes])
    st.session_state.curr_state.nodes = [node for node in st.session_state.curr_state.nodes if node.id in node_ids]
    st.session_state.curr_state.edges = [edge for edge in st.session_state.curr_state.edges if edge.source in node_ids
                                         and edge.target in node_ids]
    print([node.id for node in st.session_state.curr_state.nodes])
