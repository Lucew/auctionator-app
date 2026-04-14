import time
import io
import datetime
import logging
import re
import collections
import os
import typing

import streamlit as st
import altair as alt
import pandas as pd
import streamlit_flow as stflow

import rg_database_interactions as rgdb
import database_interactions as daint
import create_crafter_flow as ccf
import streamlit_utils as stut
import optimizer as opt
import state_serializer as stateser


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

    # reorder the columns so the name is at first
    column_names = list(styled_grouped_df)
    styled_grouped_df = styled_grouped_df[column_names[-1:] + column_names[:-1]]

    # create the gold string instead of the integers
    for column in styled_grouped_df.columns[3:-1]:
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
    regular_matched_items = set(names)
    if multiselect:
        with col2:
            regular_expression = st.text_input('Input (regular) filter expressions for the items.')
            if regular_expression:
                try:
                    regular_expression = re.compile(regular_expression)
                    regular_matched_items = set(filter(lambda x: bool(regular_expression.findall(x)), names))
                    print(regular_matched_items)
                except re.error as e:
                    st.warning(f'Regex Pattern was not valid: {str(e)}', icon="⚠️")
        with col1:

            # make the multiselect and keep the session state
            default_vals = st.session_state.get("multi_select_items", names[:1])
            default_vals = [ele for ele in default_vals if ele in regular_matched_items]
            selection = set(st.multiselect('Choose the items of interest', options=names, default=default_vals,
                                           key="multi_select_items"), )
    else:
        selection = st.selectbox('Choose the items of interest', options=names)
    return selection


def get_and_set_state():
    if "state" in st.query_params:
        param_dict = st.query_params["state"]
        state_decoded = stateser.decode_state(param_dict)
        for key, val in state_decoded.items():
            st.session_state[key] = val


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
    search = st.sidebar.text_input("Search for items").capitalize()
    df = cached_get_items_df()
    st.sidebar.dataframe(df[(df["Name (en)"].str.contains(search)) | (df["Name (de)"].str.contains(search))],
                         height=200, use_container_width=True, column_config=cl_config)

    # make a button to invalidate the cache
    st.sidebar.header('Page Settings')
    if st.sidebar.button('Invalidate Cache'):
        st.cache_data.clear()
        st.cache_resource.clear()

    st.sidebar.header('Session State (Experimental)')
    col1, col2, col3 = st.sidebar.columns(3)
    # save the session state into the url
    if col1.button('Save State'):
        state = st.session_state.to_dict()
        encoded = stateser.encode_state(state)
        st.query_params.from_dict({"state": encoded})

    # get the state from the url
    if col2.button('Read State'):
        # get the query parameters
        get_and_set_state()

    # get the state from the url
    if col3.button('Reset State'):
        # reset the params
        st.query_params.from_dict(dict())

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
                                 step=datetime.timedelta(days=1), value=(min_date, max_date+datetime.timedelta(days=1)),
                                 key='date-range-slider')

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
    apply_selection = st.checkbox('Apply Selection', value=False, key='apply-selection-checkbox')

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
        with st.expander("File Download"):

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


@st.fragment
def flow_fragment():
    """
    This function allows us to prevent redraws of the page if the user interacts only with the graph.
    :return:
    """
    st.session_state.curr_flow = stflow.streamlit_flow('static_flow', st.session_state.curr_state,
                                                       show_controls=False, fit_view=True,
                                                       show_minimap=True, hide_watermark=True,
                                                       layout=stflow.layouts.TreeLayout(direction='right'))


def reset_flow_graph():
    """
    Redraw the graph and close the toggle.
    :return:
    """
    st.session_state.curr_state_id = None
    reset_button()


def update_item_selection(update_list: typing.Iterable[str]):
    current = st.session_state.available_selection
    update_set = set(update_list)
    not_yet_select = update_set - set(current)
    current.extend(not_yet_select)
    st.session_state.available_selection = current


def crafter_page():
    logger = logging.getLogger('auctionator')

    # make a title
    st.title('Crafter')

    # get all the items we have
    items = cached_get_items()
    spells, professions = cached_get_spells()

    # create the field to put in items
    # get the items and the dataframe to map the id
    _, names, grouped_df, _ = cached_get_price_info()

    # get the item selection
    selection = st.number_input('Input the id of the item you want to craft (e.g., 49906)', 0, max(items.keys()), 49906,
                                on_change=reset_button, key='crafter-item-id-selection')

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
                                                   on_change=reset_button, key='allowed-professions-selection'))

        # create a number input for the profession level
        profession_skill = dict()
        for profession in allowed_professions:
            profession_skill[profession] = col1.number_input(f'{profession} - Maximum Skill', 0, 450, 450,
                                                             on_change=reset_button, key=f'profession-{profession}-skill-selection')

        # make a number input how much cooldown any spell can have
        maximum_cooldown = col2.number_input('Maximum cooldown (s)', 0, 1_000_000, 100_000, on_change=reset_button,
                                             key='maximum-cooldown-selection')

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

    # prune the graph after the configuration of available crafts and cooldowns
    deleted_children = prune_graph(root, allowed_professions, maximum_cooldown, profession_skill, excluded_items,
                                   excluded_spells, excluded_strings)

    # get the reference list
    reference_list = root.get_reference_list()
    necessary_items = set(node.name for nodelist in reference_list.values() for node in nodelist if isinstance(node, rgdb.ItemNode))

    # make a selection to specify available items
    with st.expander("Available Items", expanded=False):
        # get the item selection
        st.write("Select the items you have available.")
        available_selection = st.multiselect("Item IDs", names, on_change=reset_flow_graph,
                                             key='available_selection')

        # make some columns
        cols = st.columns(3)

        # create the inputs per item
        item_numbers = dict()
        for idx, item in enumerate(available_selection):
            # get the current column
            currcol = cols[idx % 3]

            # workaround for hc/non hc items with double id
            item_id = grouped_df.loc[item, 'Id']
            if isinstance(item_id, pd.Series):
                item_id = item_id.min()

            # get the id
            # TODO: Schulterplatten des tobenden Ungetüms sind doppelte id (hc, nicht hc)
            item_id = int(item_id)

            # make a text and a number select
            item_numbers[item_id] = currcol.number_input(f"{item} - Number", min_value=0, value=0,
                                                         on_change=reset_flow_graph, key=f"{item} - Number")

        # add a button that adds the items that occur in the current graph
        st.button("Input Graph Items", on_click=lambda: update_item_selection(necessary_items))

    # request the rising gods database for the crafting graph
    with st.spinner('Creating the Graph...'):

        # get the prices for the reference list
        recent_prices = daint.get_most_recent_item_price()
        recent_prices = collections.defaultdict(lambda: float('inf'), recent_prices)

        # get the cheapest combinations
        __ts = time.perf_counter()
        best_options = opt.k_best_crafting_paths(
            root=root,
            recent_prices=recent_prices,
            available_items=item_numbers,  # user inventory
            k=3,
        )

        # go through the nodes that are in the path and mark them
        if best_options[0][0] != float('inf'):
            for __node in best_options[0][2]:
                __node.mark()

        # print the flow to the page
        if st.toggle("Show the graph", key='Show-Graph-Button'):

            # check whether we already have the flow
            if "curr_state_id" in st.session_state and root.id != st.session_state.curr_state_id:
                # make the flow graph using the cheapest path
                st.session_state.curr_state, _ = ccf.create_flow(root, recent_prices, spells)
                st.session_state.curr_state_id = root.id

            # run the flow graph in a fragment
            flow_fragment()
        else:
            st.session_state.curr_state = None
            st.session_state.curr_state_id = None


        for pdx, (cheap_price, cheap_combo, best_path, remaining) in enumerate(best_options):

            # write out all the option
            link_list = [f"{root.base_url}/?item={ele}" for ele, _ in cheap_combo]
            if cheap_price == float('inf'):
                st.warning(f'We did not have a price for any valid path for {root.id}', icon="⚠️")
            else:
                if pdx == 0:
                    st.write('Best crafting path:')
                else:
                    st.write(f'Crafting path #{pdx}:')
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
