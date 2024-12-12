import collections
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout

import requestCraftDB as rcd
import parseFile as pf


def create_content_str(item: rcd.BaseNode, recent_prices: collections.defaultdict[int: [int | float]],
                       spells: dict[int: (str, str, int, str, str)]):
    if isinstance(item, rcd.ItemNode):
        price = recent_prices[item.id]
        content_str = f"""
Item {item.name}

Id: {item.id}
Amount: ({item.required_amount})

Price: {"?" if price == float("inf") else pf.price2gold(price*item.required_amount)}
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


def create_flow(root: rcd.ItemNode, recent_prices: collections.defaultdict[int: [int | float]],
                spells: dict[int: (str, str, int, str, str)]):
    # spells has the following format:
    # spell.name_en, spell.name_de, spell.cooldown, spell.profession_name, spell.skill

    # traverse the graph using layer wise bfs and create the nodes and edges
    flow_node = StreamlitFlowNode(id='0.0', pos=(0, 0),
                                  data={'content': create_content_str(root, recent_prices, spells)},
                                  node_type='input', source_position='right', target_position='left')
    layer = [(root, flow_node)]
    nodes = [flow_node]
    edges = []
    level = 1
    while layer:

        # go through the stack which corresponds to the current layer
        children = []
        for node, flnode in layer:
            for child in node.children.values():

                # create a new node
                flow_node = StreamlitFlowNode(id=f'{level+1}.{len(children)}', pos=(level*300, 50*len(children)),
                                              data={'content': create_content_str(child, recent_prices, spells)},
                                              node_type='default', source_position='right', target_position='left')

                # create a new edge
                flow_edge = StreamlitFlowEdge(f'{flnode.id}-{flow_node.id}', str(flnode.id), str(flow_node.id),
                                              animated=True, marker_end={'type': 'arrow'})

                # append both to the nodes and edges
                edges.append(flow_edge)
                nodes.append(flow_node)

                # append the child to the new level
                children.append((child, flow_node))

        # update the level
        level += 1
        layer = children

    state = StreamlitFlowState(nodes, edges)

    streamlit_flow('static_flow', state, show_controls=False, fit_view=True, show_minimap=True,
                   hide_watermark=True, layout=TreeLayout(direction='right'))
