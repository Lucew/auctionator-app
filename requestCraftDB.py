# this script can be used to get the Item DB and Crafting DB of Rising Gods
# https://db.rising-gods.de/?items for items
# https://db.rising-gods.de/?spells for spells
import requests
import re
from bs4 import BeautifulSoup


class BaseNode:
    base_url: str = "https://db.rising-gods.de"
    query_parameter: str

    def __init__(self, base_id: int, required_amount: int, parent: 'BaseNode' = None, name: str = ""):

        # save the parameters
        self.parent = parent
        self.id = base_id
        self.required_amount = required_amount
        self.children: dict[int: 'BaseNode'] = dict()
        self.name = name

    @property
    def url(self):
        return f"{self.base_url}?{self.query_parameter}={self.id}"

    def get_reference_list(self):

        # go up to the root node of the current graph
        curr = self
        while curr.parent:
            curr = curr.parent
        root = curr

        # make stack dfs and create the reference list
        stack = [root]
        reference_list = {root.id: root}
        while stack:

            # get the children
            curr = stack.pop()

            # save all the children
            reference_list.update((child.id, child) for child in curr.children.values())

            # update the stack
            stack.extend(curr.children.values())
        return reference_list

    def __str__(self):
        return f'{self.__class__} id={self.id} number={self.required_amount} {self.name if self.name else ""}'

    def print_subtree(self, indent_width: int = 4, indent: str = ""):
        # https://stackoverflow.com/a/51920869
        print(self)
        children = list(self.children.values())
        if not children:
            return
        for child in children[:-1]:
            print(f'{indent}├{"─" * indent_width}', end="")
            child.print_subtree(indent_width, f'{indent}│{" " * indent_width}')
        print(f'{indent}└{"─" * indent_width}', end="")
        children[-1].print_subtree(indent_width, f'{indent}{" " * (indent_width+1)}')


class SpellNode(BaseNode):
    query_parameter: str = "spell"

    def __init__(self, spell_id: int, required_amount: int=1, parent: [BaseNode | None] = None):
        super().__init__(spell_id, required_amount, parent)


class ItemNode(BaseNode):
    query_parameter: str = "item"

    def __init__(self, item_id: int, required_amount: int = 1, parent: [BaseNode | None] = None):
        super().__init__(item_id, required_amount, parent)


def create_item_craft_graph(item_id: int):
    # ressources id: 49906

    # create the BaseNode for initializing the graph
    root = ItemNode(item_id=item_id, required_amount=1)

    # make a request
    recipe_request = requests.get(root.url)

    # get the spells that belong to the item
    line = [ele for ele in recipe_request.text.split('\n') if ele.strip().startswith('var _ = g_spells;')]
    assert len(line) == 1, 'Line is not long enough.'
    line = line[0]
    spells = [ele[2:-3] for ele in re.findall(r"_\[\d+\]=\{", line)]

    # go through the spells and check whether we find spells that create the item
    # if that is the case we will extract the whole crafting tree, which is shown conveniently by the RG database
    # e.g.: https://db.rising-gods.de/?spell=70566
    for spell in spells:

        # get the spell pages and look for an enum on what we need
        page_list = requests.get(f"{root.base_url}/?spell={spell}").text

        # initialize the page parser
        soup = BeautifulSoup(page_list, "html.parser")

        # check whether we have a reagent list for this spell
        rs = soup.find('table', id="reagent-list-generic")
        if rs is None:
            continue

        # go through the table rows and check which items we need for the spell
        items_needed = []
        for table_row in rs.findChildren('tr'):

            # get all the td elements as they contain the padding
            td_elements = table_row.find_all('td')

            # skip table row if there is no cell
            if not td_elements:
                continue

            # get the style of the td element (and with that the level of tabs)
            cell_style = td_elements[0].get('style')
            level = 0 if cell_style is None else int(re.findall(r"\d+", cell_style)[0])

            # get the element type and id
            element_link = table_row.find('a').get('href')
            element_type, element_id = element_link.split('=')
            element_type = element_type[1:]

            # get the number of elements
            number_elements = int(re.findall(r"\d+", table_row.text)[0])

            # save the item live
            items_needed.append((level, element_type, element_id, number_elements))

        # we found a spell that creates the item, so we need to create the child node
        curr_spell = SpellNode(spell, parent=root)
        root.children[spell] = curr_spell

        # go through the parsed craft table and register the children
        parent_stack = [(curr_spell, -1)]
        for level, element_type, element_id, number_elements in items_needed:

            # get rid of deeper levels
            while level <= parent_stack[-1][1]:
                parent_stack.pop()

            # attach to the current parent
            parent = parent_stack[-1][0]
            if element_type == 'item':
                curr_node = ItemNode(element_id, number_elements, parent)
            elif element_type == 'spell':
                curr_node = SpellNode(element_id, number_elements, parent)
            else:
                raise ValueError(f'Something is of with element_type: {element_type}.')
            parent.children[element_id] = curr_node

            # add to parent stack
            parent_stack.append((curr_node, level))
    root.print_subtree()
    return root


if __name__ == '__main__':
    create_item_craft_graph(49906)
