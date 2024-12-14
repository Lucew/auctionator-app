# this script can be used to get the Item DB and Crafting DB of Rising Gods
# https://db.rising-gods.de/?items for items
# https://db.rising-gods.de/?spells for spells
import requests
import re
import sys
import logging
import time
import collections

from bs4 import BeautifulSoup


class BaseNode:
    base_url: str = "https://db.rising-gods.de"
    query_parameter: str
    marked: bool = False

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
        reference_list = collections.defaultdict(list)
        reference_list[root.id].append(root)
        while stack:

            # get the children
            curr = stack.pop()

            # save all the children
            for child in curr.children.values():
                reference_list[child.id].append(child)

            # update the stack
            stack.extend(curr.children.values())
        return reference_list

    def set_names(self, spell_dict: dict[int: str], item_dict: dict[int: str], return_when_name: bool = False):

        # check whether the root has a name
        if return_when_name and self.name:
            return

        # go up to the root node of the current graph
        curr = self
        while curr.parent:
            curr = curr.parent
        root = curr

        # make stack dfs and create the reference list
        stack = [root]
        while stack:

            # get the children
            curr = stack.pop()

            # set the name
            name_dict = spell_dict if isinstance(curr, SpellNode) else item_dict
            if curr.id in name_dict:
                curr.name = (name_dict[curr.id][1] if name_dict[curr.id][1] else name_dict[curr.id][0])

            # update the stack
            stack.extend(curr.children.values())
        return self

    def __str__(self):
        return f'{self.__class__} id={self.id} number={self.required_amount} {self.name if self.name else ""}'

    def print_subtree(self, indent_width: int = 4, indent: str = "", buffer=None):
        if buffer is None:
            buffer = sys.stdout
        # https://stackoverflow.com/a/51920869
        print(self, file=buffer)
        children = list(self.children.values())
        if not children:
            return
        for child in children[:-1]:
            print(f'{indent}├{"─" * indent_width}', end="", file=buffer)
            child.print_subtree(indent_width, f'{indent}│{" " * indent_width}', buffer=buffer)
        print(f'{indent}└{"─" * indent_width}', end="", file=buffer)
        children[-1].print_subtree(indent_width, f'{indent}{" " * (indent_width+1)}', buffer=buffer)
        return buffer

    def __hash__(self):
        return self.url

    def mark(self):
        self.marked = True

    def unmark(self):
        self.marked = False


class SpellNode(BaseNode):
    query_parameter: str = "spell"

    def __init__(self, spell_id: int, required_amount: int = 1, parent: [BaseNode | None] = None):
        super().__init__(spell_id, required_amount, parent)


class ItemNode(BaseNode):
    query_parameter: str = "item"

    def __init__(self, item_id: int, required_amount: int = 1, parent: [BaseNode | None] = None):
        super().__init__(item_id, required_amount, parent)


def create_item_craft_graph(item_id: int):
    # ressources id: 49906

    # get the logger
    logger = logging.getLogger('auctionator')
    __ts = time.perf_counter()

    # create the BaseNode for initializing the graph
    root = ItemNode(item_id=item_id, required_amount=1)

    # make a request
    recipe_request = requests.get(f'{root.url}#created-by')
    request_count = 1

    # get the spells that belong to the item
    line_identifier = 'new Listview({"template":"spell","id":"created-by"'
    line = [ele for ele in recipe_request.text.split('\n') if ele.strip().startswith(line_identifier)]

    # check whether we can craft the item
    if not line:
        return root

    # check that we found one line
    assert len(line) == 1, 'Line is not long enough.'
    line = line[0]
    spells = [int(ele[5:-1]) for ele in re.findall(r'"id":\d+,', line)]

    # go through the spells and check whether we find spells that create the item
    # if that is the case we will extract the whole crafting tree, which is shown conveniently by the RG database
    # e.g.: https://db.rising-gods.de/?spell=70566
    for spell in spells:

        # get the spell pages and look for an enum on what we need
        page_list = requests.get(f"{root.base_url}/?spell={spell}").text
        request_count += 1

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
            element_id = int(element_id)

            # get the number of elements
            number_elements = re.findall(r"\d+", table_row.text)
            number_elements = 1 if not number_elements else int(number_elements[0])

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
    logger.info(f'Request for craft table took {time.perf_counter()-__ts:0.3f}s and we made {request_count} requests.')
    return root


def request_name(item: BaseNode, language: str = 'de'):

    # get the logger
    logger = logging.getLogger('auctionator')
    __ts = time.perf_counter()

    # make a header for a german response
    custom_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/131.0.0.0 Safari/537.36',
        'accept-language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7,it;q=0.6' if language == 'de' else 'en-US'
    }

    # request the html from the database
    page = requests.get(item.url, headers=custom_headers).text

    # initialize the page parser
    soup = BeautifulSoup(page, "html.parser")

    # get all header of level one and keep the one that contains a star
    try:
        titles = soup.find_all('title')
        assert len(titles) == 1, f'Page Contained more than one or no title for {item.url}: {titles}'
        title = titles.pop()

        # check that the page is german
        if language == 'de':
            end_str = f' - {"Gegenstand" if isinstance(item, ItemNode) else "Zauber"} - Rising Gods - WotLK Database'
            assert title.text.endswith(end_str), f'Title weird for {item.url}: {title}.'
        else:
            end_str = f' - {"Item" if isinstance(item, ItemNode) else "Spell"} - Rising Gods - WotLK Database'
            assert title.text.endswith(end_str), f'Title weird for {item.url}: {title}.'
        item_name = title.text[:-len(end_str)]

        # check whether this spell is from some tradework
        keywords = soup.find('meta', {'name': "keywords"})
        keywords = str(keywords).split('"')[1].split(', ')
        profession_name = ""
        cooldowns = 0
        skill_level = 0

        # get the infobox
        if keywords[-1] == 'Professions' or keywords[-1] == 'Berufe':

            # get the profession name
            profession_name = keywords[-2]

            # get the infobox
            infobox = str(soup.find('table', {"class": "infobox"}))

            # get the lines with the requirements for skill
            lines = [line.strip() for line in infobox.splitlines() if line.strip().startswith('Markup.printHtml("')]
            assert len(lines) == 1, 'Could not find spell specification.'
            lines = re.findall(r'\[color=r\d+]\d+\[/color]', lines[0])
            if len(lines) >= 1:
                lines = re.findall(r"]\d+\[", lines[0])
                skill_level = int(lines[0][1:-1])

            # get the cooldown if there is
            table = soup.find('table', {"class": "grid", "id": "spelldetails"})
            assert table is not None, 'Could not find the details table.'
            cooldowns = [child.text for child in table.findChildren('tr', recursive=True)
                         if 'Cooldown' in child.text or 'Abklingzeit' in child.text]
            assert len(cooldowns) == 1, 'There are more cooldown than expected.'
            cooldowns = cooldowns[0].strip()

            # check whether we have a cooldown
            if cooldowns.endswith('n/a') or cooldowns.endswith('n. v.'):
                cooldowns = 0
            else:
                _, cooldowns, unit = cooldowns.split()
                units = {'days': 86400, 'hours': 3600, 'minutes': 60, 'seconds': 1, 'Tage': 86400,
                         'Stunden': 3600, 'Minuten': 60, 'Sekunden': 1}
                cooldowns = float(cooldowns)*units[unit]

        # measure the time
        logger.info(f'Request for {"Spell" if isinstance(item, SpellNode) else "Item"} {item.id} '
                    f'(language {language}) took {time.perf_counter() - __ts:0.3f}s.')
        return item_name, profession_name, cooldowns, skill_level
    except AssertionError as e:
        logger.error(f'Request_Name error! {str(e)}')
        return None, None, None, None


def test():

    # get some names for spells and items
    request_name(ItemNode(22854))
    request_name(SpellNode(66658))
    request_name(SpellNode(28589))
    request_name(SpellNode(54222))
    request_name(SpellNode(53812))
    request_name(SpellNode(60893))

    # request not existing item
    request_name(SpellNode(0))
    request_name(ItemNode(0))

    # get a craft graph
    create_item_craft_graph(29520)
    create_item_craft_graph(49906)


if __name__ == '__main__':
    test()
