import time
import io
import datetime
import logging
import re
import collections
import os

import streamlit as st
import altair as alt
import pandas as pd
import streamlit_flow as stflow

import rg_database_interactions as rgdb
import database_interactions as daint
import create_crafter_flow as ccf
import streamlit_utils as stut


# make a cached function to get the data stuff
@st.cache_data
def cached_get_price_info():
    __ts = time.perf_counter()

    # get the dataframe and name2link converter
    df = daint.db2df(daint.get_item_prices())

    # get the names of all items we have prices for
    names = list(df['Name'].unique())

    # create an overview per item with all prices and turn prices per item into array
    grouped_df = df.groupby(['Name', 'Id'])
    grouped_df = grouped_df['Price'].describe().join(
        grouped_df['Price'].apply(lambda x: [ele for ele in x.values]).to_frame('Prices'))

    # delete Id from the index
    grouped_df = grouped_df.reset_index(level=("Id", ))

    old_version = """
    style_format = dict()
    for name in grouped_df.columns[1:-1]:
        style_format[name] = daint.price2gold
    style_format['count'] = int

    # make a styled group dataframe, where we replace all the integers with strings so it is easier to read
    styled_grouped_df = grouped_df.copy().style.format(style_format).format_index(urllib.parse.unquote, axis=1)
    """

    # make another dataframe, where the item prices become human friendly strings
    styled_grouped_df = grouped_df.copy()

    # create the links and put them into the index
    styled_grouped_df['Name'] = [daint.name2dblink(idd, name) for idd, name in
                                 zip(styled_grouped_df['Id'], styled_grouped_df.index)]
    # create the gold string instead of the integers
    for column in styled_grouped_df.columns[2:-2]:
        styled_grouped_df[column] = styled_grouped_df[column].apply(daint.price2gold)

    # log the time it took to create the cache
    logger = logging.getLogger('auctionator')
    logger.info(f"Creating the price dataframe (requesting, grouping, and caching) took {time.perf_counter()-__ts} s.")
    return df, names, grouped_df, styled_grouped_df


@st.cache_data
def cached_get_items():
    return daint.get_items()


@st.cache_data
def cached_get_items_df():

    # get the items dict from cache
    items_dict = cached_get_items()

    # make a dataframe
    df = pd.DataFrame.from_dict(items_dict, orient='index', columns=["Name (en)", "Name (de)"])
    return df


@st.cache_resource
def cached_get_spells():
    return daint.get_spells()


# IMPORTANT: Cache the conversion to prevent computation on every rerun
@st.cache_data
def convert_df(df, typed: str):
    if typed == r'text\csv':
        return df.to_csv().encode("utf-8")

    elif typed == r'application/vnd.ms-excel':
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Auctionator')
        return buffer
    else:
        raise ValueError('Unrecognized download type!')


# cache some of the crafting data
@st.cache_data(max_entries=100)
def cached_crafting_recipe(item_id: int):
    return rgdb.create_item_craft_graph(item_id)


# cache some of the item database requests
@st.cache_data(max_entries=20, hash_funcs={rgdb.ItemNode: rgdb.BaseNode.__hash__,
                                           rgdb.SpellNode: rgdb.BaseNode.__hash__})
def cache_request_name(item: rgdb.BaseNode, language: str):
    return rgdb.request_name(item, language)


def item_selector(names: list[str], multiselect: bool = True):

    # check whether we wanna do a multiselect

    # make an item selector with regular expression filter
    col1, col2 = st.columns(2)
    regular_matched_items = []
    if multiselect:
        with col2:
            regular_expression = st.text_input('Input (regular) filter expressions for the items.')
            if regular_expression:
                try:
                    regular_expression = re.compile(regular_expression)
                    regular_matched_items = list(filter(lambda x: bool(regular_expression.findall(x)), names))
                except re.error as e:
                    st.warning(f'Regex Pattern was not valid: {str(e)}', icon="⚠️")
        with col1:

            # make the multiselect and keep the session state
            default_vals = st.session_state.get("multi_select_items", names[:1])
            default_vals = list(set(default_vals) | set(regular_matched_items))
            selection = set(st.multiselect('Choose the items of interest', options=names, default=default_vals,
                                           key="multi_select_items"), )
    else:
        selection = st.selectbox('Choose the items of interest', options=names)
    return selection


def side_bar():
    # get the logger
    logger = logging.getLogger('auctionator')

    st.sidebar.title('Navigation')
    st.sidebar.header('Application')
    choice = st.sidebar.selectbox('Choose Application', options=['Analyzer', 'Crafter', 'Extender'])

    # create a file upload
    st.sidebar.header('File Upload')
    uploaded_file = st.sidebar.file_uploader('Upload your LUA file here.')

    # To convert to a string based IO
    if uploaded_file is not None:

        # make the logging
        logger.info('A file was uploaded.')

        # get the file into a string
        stringio = uploaded_file.getvalue().decode("utf-8")

        # parse the string
        success_val, succes_str, skipped_items = daint.write_data(stringio)

        if success_val:

            # write the information to the database
            st.sidebar.write(succes_str)
            st.sidebar.write(f'Skipped {skipped_items[-1][1][0]}/{skipped_items[-1][2][0]} items due to missing id.')
            st.sidebar.write(f'Skipped {skipped_items[-1][1][1]}/{skipped_items[-1][2][1]} items due to missing name.')

            # save the file onto the disc
            file_path = os.path.join(os.getcwd(), 'save', f'{time.time()}_auctionator.lua')
            if not os.path.isfile(file_path):
                with open(file_path, 'w') as filet:
                    filet.write(stringio)
        else:
            st.sidebar.warning(succes_str, icon="⚠️")

        # clean the function cache
        if success_val:
            # https://stackoverflow.com/a/77676594
            cached_get_price_info.clear()
            logger.info('File is parsed and cache invalidated.')

    # make a dataframe out of the items
    cl_config = {"_index": st.column_config.NumberColumn("Id", format="%d")}
    st.sidebar.header('Item Finder')
    search = st.sidebar.text_input("Search for items")
    df = cached_get_items_df()
    st.sidebar.dataframe(df[(df["Name (en)"].str.contains(search)) | (df["Name (de)"].str.contains(search))],
                         height=100, use_container_width=True, column_config=cl_config)

    # make a button to invalidate the cache
    st.sidebar.header('Page Settings')
    if st.sidebar.button('Invalidate Cache'):
        st.cache_data.clear()
        st.cache_resource.clear()

    # put my name on the page
    styl = f"""  
    <div style="position: relative">
            <p style="position: fixed; bottom: 0; text-align: center"> Made with 💝 by Lucas.
            </p>
    </div>
    """
    st.sidebar.markdown(styl, unsafe_allow_html=True)
    return choice


def analyzer_page():

    # get the logger
    logger = logging.getLogger('auctionator')

    # create the header and initial explanation
    st.title('WoW Auctionator Analyzer')
    st.write('This app analyzes your auctionator.lua (WoW 3.3.5a) file and writes the item prices to a database.'
             ' Using this information one can analyze long term item prices and some statistical properties.'
             ' You can download all collected data on the left. This is a private project. Please use with caution!')

    # get the dataframe
    df, names, grouped_df, styled_grouped_df = cached_get_price_info()

    # get the selection
    selection = item_selector(names)

    # create the selection within the dataframe
    selected_df = df
    if selection:

        # select the items from the dataframe
        selected_df = df[df['Name'].isin(selection)]
        min_date = selected_df['Date'].min().to_pydatetime()
        max_date = selected_df['Date'].max().to_pydatetime()

        # make a date selection
        time1, time2 = st.slider('Date Range', min_date, max_date+datetime.timedelta(days=1),
                                 step=datetime.timedelta(days=1), value=(min_date, max_date+datetime.timedelta(days=1)))

        # apply the time selection
        selected_df = selected_df[(pd.Timestamp(time1) <= selected_df['Date']) &
                                  (selected_df['Date'] <= pd.Timestamp(time2))]

        # make the line chart
        c = alt.Chart(selected_df).mark_line(point=True).encode(x="Date", y="Price",
                                                                  color=alt.Color("Name").legend(orient="bottom",
                                                                                                 labelLimit=0,
                                                                                                 symbolLimit=0,
                                                                                                 columns=4),
                                                                  tooltip=["Date", "Name", 'Gold'])
        st.altair_chart(c, use_container_width=True)

    # style the dataframe
    cl_config = {"Prices": st.column_config.LineChartColumn("Prices"),
                 "Name": st.column_config.LinkColumn('Name', display_text=r"[?&]name=([^&#]+)$"),
                 "Id": st.column_config.NumberColumn(format="%d"),
                 "Count": st.column_config.NumberColumn(format="%d")}

    # make a checkbox whether to apply selection
    apply_selection = st.checkbox('Apply Selection', value=False)

    # display the dataframe
    if apply_selection:
        display_df = styled_grouped_df.loc[list(selection)]
    else:
        display_df = styled_grouped_df

    # apply some filters
    display_df = stut.filter_dataframe(display_df)

    # visualize the dataframe
    st.write(f'Description of Price Distribution for {"the selected" if apply_selection else "all"}'
             f' Items (click columns to sort):')
    st.dataframe(display_df, column_config=cl_config, hide_index=True, use_container_width=True)

    # make a sidebar
    with st.sidebar:

        st.sidebar.header('File Download')

        # make a selection for the sidebar
        download_type = st.selectbox('Choose the download type', options=['All', 'Selection'])

        # make the dataframe selection
        if download_type == 'All':
            curr_df = df
        elif download_type == 'Selection':
            curr_df = selected_df
        else:
            raise ValueError('Unspecified download!')

        # make a csv download button
        cs_dwn = st.download_button(
            label="Download data as CSV",
            data=convert_df(curr_df, r'text\csv'),
            file_name=f"{'selected_' if download_type == 'Selection' else ''}data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if cs_dwn:
            logger.info(f'Download CSV-File {"(selection)" if download_type == "Selection" else ""}.')

        # Excel download button
        # MIME type: https://stackoverflow.com/a/1964182
        exc_down = st.download_button(
            label="Download data as XLSX",
            data=convert_df(curr_df, r'application/vnd.ms-excel'),
            file_name=f"{'selected_' if download_type == 'Selection' else ''}data.xlsx",
            mime="text/csv",
            use_container_width=True,
        )
        if exc_down:
            logger.info(f'Download XLSX-File {"(selection)" if download_type == "Selection" else ""}.')


def prune_graph(graph: rgdb.BaseNode, allowed_professions: set[str], maximum_cooldown: int,
                skill_per_profession: dict[str:int], excluded_items: set[int], excluded_spells: set[int],
                excluded_strings: set[str]):

    # get the spells
    spells, _ = cached_get_spells()

    # check that we got a root
    assert graph.is_root(), 'Can only prune from root.'
    assert isinstance(graph, rgdb.ItemNode), 'Root is a spell.'

    # go through the nodes and make the checks
    stack = [graph]
    deleted_children = []
    while stack:

        # get the current node
        node = stack.pop()

        # make the checks
        child_keys = list(node.children.keys())
        for child_key in child_keys:

            # get the child
            child = node.children[child_key]

            # check for the names
            if child.name and child.name in excluded_strings:
                deleted_children.append(child)
                del node.children[child_key]
                continue

            # check if it is an item
            if isinstance(child, rgdb.ItemNode):
                if child.id in excluded_items:
                    deleted_children.append(child)
                    del node.children[child_key]
                else:
                    stack.append(node.children[child_key])
                continue

            # check whether the spell is allowed
            if child.id in excluded_spells:
                deleted_children.append(child)
                del node.children[child_key]
                continue

            # guard clause to check whether we have information about the spell
            if child.id not in spells:
                stack.append(child)
                continue

            # get the spell information
            name_en, name_de, cooldown, profession_name, skill = spells[child.id]

            if profession_name not in allowed_professions or cooldown > maximum_cooldown or \
                    skill > skill_per_profession[profession_name]:
                # we need to prune this child
                deleted_children.append(child)
                del node.children[child_key]
            else:
                stack.append(node.children[child_key])

    return deleted_children


def reset_button():
    st.session_state['Show-Graph-Button'] = False


def crafter_page():
    logger = logging.getLogger('auctionator')

    # make a title
    st.title('Crafter')

    # get all the items we have
    items = cached_get_items()
    spells, professions = cached_get_spells()

    # get the item selection
    selection = st.number_input('Input the id of the item you want to craft (e.g., 49906)', 0, max(items.keys()), 49906,
                                on_change=reset_button)

    # Write the item
    if selection not in items:
        st.warning(f'Item with id {selection} not found.', icon="⚠️")
        return dict()

    # write the name of the item
    st.header(f'{" ".join(ele.capitalize() for ele in items[selection][0].split())} - {selection}')
    item = rgdb.ItemNode(selection)
    st.markdown(f"[{items[selection][0]}]({item.url})")

    # request the rising gods database for the crafting graph
    with st.spinner('Requesting Craft Graph...'):

        # get the recipe from the database (4096 is a problem)
        root = cached_crafting_recipe(selection)

        # set the names of the spells
        root.set_names(spells, items, return_when_name=True)

    # make the configuration (and reset the graph button along the way
    with st.expander("Configuration"):

        # create some columns
        col1, col2 = st.columns(2)

        # make a multiselect for the allowed profession
        profession_list = list(set(spells[spell.id][3] for spell in root.dfs(target_class=rgdb.SpellNode)
                                   if spell.id in spells and spells[spell.id][3]))
        allowed_professions = set(col1.multiselect('Select allowed professions', profession_list, profession_list,
                                                   on_change=reset_button))

        # create a number input for the profession level
        profession_skill = dict()
        for profession in allowed_professions:
            profession_skill[profession] = col1.number_input(f'{profession} - Maximum Skill', 0, 450, 450,
                                                             on_change=reset_button)

        # make a number input how much cooldown any spell can have
        maximum_cooldown = col2.number_input('Maximum cooldown (s)', 0, 1_000_000, 100_000, on_change=reset_button)

        # create a multiselect excluded spells
        options = set(spell.id for spell in root.dfs(target_class=rgdb.SpellNode))
        excluded_spells = set(col2.multiselect('Exclude Spells', options, on_change=reset_button))

        # create a multiselect for excluded items
        options = set(item.id for item in root.dfs(target_class=rgdb.ItemNode) if not item.is_root())
        excluded_items = set(col2.multiselect('Exclude Items', options, on_change=reset_button))

        # make a regular expression matching
        regular_expression = col2.text_input('Input (regular) filter expressions for names.', on_change=reset_button)
        options = set(spell.name for spell in root.dfs(target_class=rgdb.SpellNode) if spell.name)
        regular_matched_items = []
        if regular_expression:
            try:
                regular_expression = re.compile(regular_expression)
                regular_matched_items = list(filter(lambda x: bool(regular_expression.findall(x)), list(options)))
            except re.error as e:
                st.warning(f'Regex Pattern was not valid: {str(e)}', icon="⚠️")

        # create a string for excluded items
        default_vals = st.session_state.get('crafter_multi_exclude', [])
        default_vals = list(set(default_vals) | set(regular_matched_items))
        excluded_strings = set(col2.multiselect('Exclude Names', options, key='crafter_multi_exclude',
                                                default=default_vals, on_change=reset_button))

    # request the rising gods database for the crafting graph
    with st.spinner('Creating the Graph...'):

        # prune the graph
        deleted_children = prune_graph(root, allowed_professions, maximum_cooldown, profession_skill, excluded_items,
                                       excluded_spells, excluded_strings)

        # get the reference list
        reference_list = root.get_reference_list()

        # get the prices for the reference list
        recent_prices = daint.get_most_recent_item_price()
        recent_prices = collections.defaultdict(lambda: float('inf'), recent_prices)

        # make dfs through the item tree
        def dfs(node):
            # we reached a leaf
            if len(node.children) == 0:
                assert isinstance(node, rgdb.ItemNode), 'Something is off.'

                # every dfs returns the current price, the current items that are necessary and the path it took
                return recent_prices[node.id]*node.required_amount, [(node.id, node.required_amount)], [node]

            # if we are a spell node, we need to sum the items we use
            if isinstance(node, rgdb.SpellNode):
                # get all the items we need
                curr_price = [dfs(child) for child in node.children.values()]

                # fuse the items together
                item_dict = collections.defaultdict(int)
                tmp_price = 0
                tmp_path = []
                for pprice, item_path, path in curr_price:
                    for __item, number in item_path:
                        item_dict[__item] += number
                    tmp_path.extend(path)
                    tmp_price += pprice

                # append the own spell
                tmp_path.append(node)

                # get the path again
                curr_price = (tmp_price, list(item_dict.items()), tmp_path)

            # if we are an item node, we need to find the cheapest option
            elif isinstance(node, rgdb.ItemNode):

                # check the options to craft the item
                curr_price = min((dfs(child) for child in node.children.values()), key=lambda x: x[0])

                # check the option to just take the item itself
                own_price = recent_prices[node.id]*node.required_amount
                if own_price < curr_price[0]:
                    curr_price = (own_price, [(node.id, node.required_amount)], [node])
            else:
                raise ValueError('Something is off.')
            return curr_price
        # get the cheapest combination
        __ts = time.perf_counter()
        cheap_price, cheap_combo, best_path = dfs(root)

        # go through the nodes that are in the path and mark them
        if cheap_price != float('inf'):
            for __node in best_path:
                __node.mark()

        # print the flow to the page
        if st.toggle("Show the graph", key='Show-Graph-Button'):
            state, _ = ccf.create_flow(root, recent_prices, spells)
            st.session_state.curr_state = stflow.streamlit_flow('static_flow', state,
                                                                show_controls=False, fit_view=True,
                                                                show_minimap=True, hide_watermark=True,
                                                                layout=stflow.layouts.TreeLayout(direction='right'))

        # write out all the option
        link_list = [f"{root.base_url}/?item={ele}" for ele, _ in cheap_combo]
        if cheap_price == float('inf'):
            st.warning(f'We did not have a price for any valid path for {root.id}', icon="⚠️")
        else:
            st.write('Best crafting path:')
            st.markdown(daint.price2gold(cheap_price) +
                        "-".join(f'[{items[ele][0]}]({linked}) ({number})'
                                 for (ele, number), linked in zip(cheap_combo, link_list)))
        logger.info(f'Searching the graph for item {root.id} (name={items.get(root.id, ("NOT FOUND", ""))[0]}) '
                    f'took {time.perf_counter() - __ts:0.4f}s.')
    return reference_list


def extender_page():

    # make a title
    st.title('Extender')
    st.write('This allows to extend item names and spell names.')

    # select whether to update item or spell
    update_type = st.selectbox('What do you want to update?', ['Spell names', 'Item names'])

    # decide what to do
    if update_type == 'Item names':
        extend_item()
    elif update_type == 'Spell names':
        extend_spell()


def extend_item():
    # get all the items we have
    items = cached_get_items()

    # enter some item id
    curr_item = st.number_input('Input the id of the item you want to change (e.g., 49906)', 0, max(items.keys()),
                                49906)

    # search the item in our database
    if curr_item not in items:
        st.warning(f'item not in our database: {curr_item}', icon="⚠️")
        return

    # request name from the db
    item = rgdb.ItemNode(curr_item)
    st.markdown(f'Found: [{items[curr_item][0]}]({item.url}) (german name: [{items[curr_item][1]}]({item.url}))')
    st.divider()
    # get the name from the database
    with st.spinner('Requesting Name...'):
        item_name = cache_request_name(item, language='de')[0]

    # update the name
    if item_name is not None:
        st.markdown(f'We found the german name for this item id={item.id}: [{item_name}]({item.url}).')
        st.write('Shall we update the german name in our database?')
        updater = st.button('Update')
        if updater:
            # update the database
            daint.update_db_item_name_de(item.id, item_name)
            st.write('✅')

            # invalidate the items cache
            cached_get_items.clear()


def extend_spell():

    # enter some spell id
    curr_spell = st.number_input('Input the id of the spell you want to change (e.g., 70566)', 0, 1_000_000, 70566)

    # get all the items we have
    spells, _ = cached_get_spells()

    # create the corresponding item
    spell = rgdb.SpellNode(curr_spell)

    # search the item in our database
    if curr_spell not in spells:
        st.warning(f'Spell not in our database: {curr_spell}', icon="⚠️")
    else:
        st.markdown(f'Found: [{spells[curr_spell][0]}]({spell.url}) '
                    f'(german name: [{spells[curr_spell][1]}]({spell.url}))')
    st.divider()

    # get the name from the database
    with st.spinner('Requesting Name...'):
        spell_name, profession_name, cooldown, skill_level = cache_request_name(spell, language='de')
        spell_name_en, profession_name_en, _, _ = cache_request_name(spell, language='en')

    # update the name
    if spell_name is not None:
        st.markdown(f'We found the german name for this spell id={spell.id}: [{spell_name}]({spell.url}).')
        st.markdown(f'We found the english name for this spell id={spell.id}: [{spell_name_en}]({spell.url}).')
        st.write('Shall we update the name in our database?')
        updater = st.button('Update')
        if updater:

            # update the database
            daint.insert_db_spell_name(spell.id, spell_name, spell_name_en,
                                       profession_name, profession_name_en, cooldown, skill_level)
            st.write('✅')

            # invalidate the spells cache
            cached_get_spells.clear()
    else:
        st.markdown(f'We did not find spell: [{spell.id}]({spell.url}).')


def update_spells_from_reference_list(reference_list: dict[int: list[rgdb.BaseNode]], sleep_time: float = 0.4):

    # check for unknown spells
    spells, _ = cached_get_spells()
    unknown_spells = [rgdb.SpellNode(spell_id) for spell_id, node_list in reference_list.items()
                      if node_list and isinstance(node_list[0], rgdb.SpellNode) and spell_id not in spells]

    # check whether there are unknown spells
    if not unknown_spells:
        return

    # create a button to update all spells
    st.write(f'Updating will take ~{int(len(unknown_spells)*(sleep_time+0.1)+1)}s')
    update_all_spells = st.button('Update Spells.', on_click=reset_button)
    if not update_all_spells:
        return
    with st.spinner('Updating spell names...'):
        for spell in unknown_spells:
            spell_name, profession_name, cooldown, skill_level = cache_request_name(spell, language='de')
            spell_name_en, profession_name_en, _, _ = cache_request_name(spell, language='en')
            if spell_name and spell_name_en:
                daint.insert_db_spell_name(spell.id, spell_name, spell_name_en,
                                           profession_name, profession_name_en, cooldown, skill_level)
            time.sleep(sleep_time)

    # invalidate the cache
    cached_get_spells.clear()


def update_items_from_reference_list(reference_list: dict[int: list[rgdb.BaseNode]], sleep_time: float = 0.4):

    # check for unknown spells
    items = cached_get_items()
    unknown_items = [rgdb.ItemNode(item_id) for item_id, node_list in reference_list.items()
                     if node_list and isinstance(node_list[0], rgdb.ItemNode) and not items[item_id][1]]

    # check whether there are unknown items
    if not unknown_items:
        return

    # create a button to update all spells
    st.write(f'Updating will take ~{int(len(unknown_items)*(sleep_time+0.1)+1)}s')
    update_all_spells = st.button('Update Items.', on_click=reset_button)
    if not update_all_spells:
        return
    with st.spinner('Updating item names...'):
        for item in unknown_items:

            # get the name from the database
            item_name = cache_request_name(item, language='de')[0]
            time.sleep(sleep_time)

            # update the database
            daint.update_db_item_name_de(item.id, item_name)

    # invalidate the items cache
    cached_get_items.clear()


# set the layout to wide
st.set_page_config(layout="wide", page_title="Autionator")

# get the logger
stut.init_logging()

# make the page
__choice = side_bar()
if __choice == 'Analyzer':
    analyzer_page()
elif __choice == 'Crafter':
    __ref_list = crafter_page()
    update_spells_from_reference_list(__ref_list)
    update_items_from_reference_list(__ref_list)
elif __choice == 'Extender':
    extender_page()
