import numpy as np
from slpp import slpp as lua
import os
import re
import time
import datetime
import sqlalchemy
from createDatabase import Item, Price, Base
import collections
import pygsheets
import pandas as pd
import urllib.parse
import logging


def parse_data(content: str = None):
    __ts = time.perf_counter()

    if content is None:
        # read the file into memory
        with open(os.path.join('ressources', 'Auctionator.lua'), encoding='UTF-8') as filet:
            content = filet.read()

    # get the starting time für auctionator data
    # https://github.com/alchem1ster/WotLK-Auctionator/blob/b4e3eb68dcc7cc1142b73d8a978d471bf0d4110a/Auctionator/Auctionator.lua#L503
    __startdate = datetime.datetime.strptime('2008-08-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp()

    # go through the text and check where we reach level zero curly bracers, so we know where to cut the different
    # variables (using curly bracer level
    curr_level = 0
    last_index = 0
    variable_dict = dict()
    for ele in re.finditer(r'[{}]', content):

        # get the position of the found element
        px = ele.start()

        # check at which level we are
        curr_level += 1 if content[px] == '{' else -1

        # make a guard clause so we only progress when we reach level zero of curly bracers
        if curr_level:
            continue

        # get the current string that contains a variable
        curr_variable = content[last_index:px+1]

        # split into variable name and the dict itself
        variable_name, lua_dict = curr_variable.split(' = ', 1)

        # parse the lua dict
        lua_dict = lua.decode(lua_dict)

        # clean the variable name
        variable_name = variable_name.strip()

        # put the variables into the dict
        variable_dict[variable_name] = lua_dict

        # update the strings we have found
        last_index = px+1

    # translate the dates in the history
    history = variable_dict['AUCTIONATOR_PRICING_HISTORY']
    for item, value_dict in history.items():

        # make the new dict by converting the timestamps and splitting the : format
        value_dict = {(convert_auctionator_time(int(ele.split(':')[0]), __startdate),
                       ele.split(':')[1] if len(ele.split(':')) > 1 else '')
                      if ele != 'is' else ele:
                          tuple(float(ele) for ele in val.split(':'))
                      for ele, val in value_dict.items()}

        # explanation of the tag type
        # https://github.com/alchem1ster/WotLK-Auctionator/blob/b4e3eb68dcc7cc1142b73d8a978d471bf0d4110a/Auctionator/Auctionator.lua#L2828

        # sanity check that every item has at least an id
        assert 'is' in value_dict, f'Item {item} has no id.'

        # replace the old dictionary
        history[item] = value_dict

    logger = logging.getLogger('auctionator')
    logger.info(f'Parsing took {time.perf_counter()-__ts: 0.2f}s')
    return history


def convert_auctionator_time(time_int:int, zero_time: float):
    # https://github.com/alchem1ster/WotLK-Auctionator/blob/b4e3eb68dcc7cc1142b73d8a978d471bf0d4110a/Auctionator/Auctionator.lua#L4486
    return (time_int*60) + zero_time


def pretty_print_hist(hist: dict[tuple[int|str, str]: tuple[int, int]]):
    for (date, typed), (price, count) in hist.items():
        if date != 'i':
            print(datetime.datetime.fromtimestamp(date), typed, price, count)


def write_to_database(hist: dict[tuple[int|str, str]: tuple[int, int]]):
    __ts = time.perf_counter()
    logger = logging.getLogger('auctionator')
    db = sqlalchemy.create_engine('sqlite:///auctionator.db')
    session = sqlalchemy.orm.Session(db)
    Base.metadata.create_all(db)

    # go through the data and create items if necessary, otherwise check that id is unique
    for item, value_dict in hist.items():

        # get the current id
        curr_id = int(value_dict['is'][0])

        # check whether we already have the id or the name
        check_query = session.query(Item).filter_by(item_id=curr_id).all()
        if check_query:
            try:
                assert check_query[0].item_id == curr_id, f'Something with {item}:{curr_id} is off.'

                # check whether we are missing german names in the database and the name is german
                # then we update the name
                for itt in check_query:
                    if itt.name_de == "" and item != itt.name:
                        itt.name_de = item
            except AssertionError as e:
                logger.error(f'Logging error! AssertionError --> {str(e)}')
                return False, f'Invalid Items in File (english client?). Error with Item {item}, Id={curr_id}.'
        else:
            error_str = f'Invalid Items in File - Id Unknown! Error with Item {item}, Id={curr_id}.'
            logger.error(f'Logging error! --> {error_str}')
            return False, error_str

        for typed, stacks in value_dict.items():

            # skip the id
            if typed == 'is':
                continue

            # skip all mean values
            if typed[1]:
                continue

            # check whether this item with this date is already there
            check_query = session.query(Price).filter_by(id=curr_id, unix_timestamp=typed[0]).one_or_none()
            if check_query is None:
                session.add(Price(id=curr_id, unix_timestamp=typed[0], price=stacks[0], stacks=int(stacks[1])))
    session.commit()
    logger.info(f'Writing to DB took {time.perf_counter() - __ts: 0.2f}s')
    return True, 'Success!'


def get_item_prices():
    __ts = time.perf_counter()
    db = sqlalchemy.create_engine('sqlite:///auctionator.db')
    session = sqlalchemy.orm.Session(db)
    Base.metadata.create_all(db)
    grouped_prices = session.query(Price, Item).filter(Price.id == Item.item_id).all()
    prices_dict = collections.defaultdict(list)
    for group in grouped_prices:
        prices_dict[(group[1].item_id, group[1].name_de if group[1].name_de else group[1].name)].append(
            (group[0].unix_timestamp, group[0].price/group[0].stacks))
    for idx, val in prices_dict.items():
        val.sort()
    logger = logging.getLogger('auctionator')
    logger.info(f'Querying the DB for prices took {time.perf_counter() - __ts: 0.2f}s')
    return prices_dict


def get_items():
    __ts = time.perf_counter()
    db = sqlalchemy.create_engine('sqlite:///auctionator.db')
    session = sqlalchemy.orm.Session(db)
    Base.metadata.create_all(db)
    result = session.execute(sqlalchemy.select(Item.item_id, Item.name, Item.name_de)).all()
    item_dict = {rs.item_id: (rs.name, rs.name_de) for rs in result}
    logger = logging.getLogger('auctionator')
    logger.info(f'Querying the DB for all items took {time.perf_counter() - __ts: 0.2f}s')
    return item_dict


def get_most_recent_item_price():
    __ts = time.perf_counter()
    db = sqlalchemy.create_engine('sqlite:///auctionator.db')
    session = sqlalchemy.orm.Session(db)
    Base.metadata.create_all(db)
    recent_price = session.execute(sqlalchemy.text('SELECT i.item_id, p.price FROM prices p JOIN items i ON p.id == i.item_id WHERE p.unix_timestamp == (SELECT MIN(unix_timestamp) FROM prices p2 WHERE p2.id == p.id)')).all()
    logger = logging.getLogger('auctionator')
    logger.info(f'Querying the DB for the most recent prices took {time.perf_counter() - __ts: 0.2f}s')
    return dict(recent_price)


def write_to_gsheet(prices_dict: dict[tuple[int, str]: list[tuple[int, float]]]):
    __ts = time.perf_counter()
    # authorization
    # if you are using service account make sure to share the sheet with that account. only then you can access then
    # see account in the json file
    gc = pygsheets.authorize(service_file=os.path.join('ressources', 'auctionator-443900-c59258f330b3.json'))

    # open the google spreadsheet
    sheet = gc.open('Auctionator')[0]

    # get the item with the most dates (so we can create our matrix accordingly
    max_items = max(len(vals) for vals in prices_dict.values())

    # go through the items and write name and id and then date and price
    contents = [['']*(max_items+1) for _ in range(len(prices_dict)*3+1)]
    for rx, ((idd, name), sorted_prices) in enumerate(prices_dict.items()):
        rx *= 3

        # write the name and the dates in the first row
        contents[rx][0] = name
        contents[rx+1][0] = idd
        for idx, (date, price) in enumerate(sorted_prices, 1):
            contents[rx][idx] = str(datetime.datetime.fromtimestamp(date))
            contents[rx+1][idx] = price
    ranged = f'{get_cell_name(0, 0)}:{get_cell_name(len(prices_dict)*3, max_items)}'
    sheet.update_values(ranged, contents)
    logger = logging.getLogger('auctionator')
    logger.info(f'Writing to sheet took {time.perf_counter() - __ts: 0.2f}s')


def get_cell_name(row: int, col: int) -> str:
    """
    Maps row and column indices to Excel-style cell names.

    Parameters:
        row (int): The row index (0-based).
        col (int): The column index (0-based).

    Returns:
        str: The Excel-style cell name.
    """
    # Convert column index to letters
    column_name = ''
    col_index = col + 1  # Convert to 1-based index for Excel
    while col_index > 0:
        col_index -= 1
        column_name = chr(col_index % 26 + 65) + column_name
        col_index //= 26

    # Convert row index to 1-based row number
    row_number = row + 1

    # Combine column letters with row number
    return f"{column_name}{row_number}"


def write_data(input_str: str = None):
    logger = logging.getLogger('auctionator')
    try:
        history = parse_data(input_str)
    except ValueError as e:
        logger.error(f'Parsing error: ValueError --> {str(e)}.')
        return False, 'Invalid Lua file.'
    write_success = write_to_database(history)
    # write_to_gsheet(prices_dict)
    return write_success


def price2gold(price: float):
    # check that price is not None
    # https://stackoverflow.com/a/944712
    if not (isinstance(price, float), isinstance(price, int)) or price != price:
        return price

    # compute gold etc. from price
    mult = 100
    gold = [0, 0, 0]
    price = int(price)
    for idx in range(2):
        price, rest = divmod(price, mult)
        gold[idx] = int(rest)
    gold[-1] = int(price)
    return f'{gold[-1]}🥇 {gold[-2]:02}🥈 {gold[-3]:02}🥉'


def db2df(history_dict: dict[tuple[int, str]: list[tuple[int, float]]]):
    # create a dataframe for plotting
    df_dict = {'Name': [], 'Price': [], 'Date': [], 'Gold': [], 'Id': []}
    name2link = dict()
    for ((idd, name), sorted_prices) in history_dict.items():
        df_dict['Name'].extend(name for _ in range(len(sorted_prices)))
        df_dict['Price'].extend(price for _, price in sorted_prices)
        df_dict['Date'].extend(date for date, _ in sorted_prices)
        df_dict['Gold'].extend(price2gold(price) for _, price in sorted_prices)
        df_dict['Id'].extend(idd for _ in range(len(sorted_prices)))
        name2link[name] = name2dblink(idd, name)
    df = pd.DataFrame(df_dict)
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    return df, name2link


def name2dblink(item_id: int, name: str):
    return f'https://db.rising-gods.de/?item={item_id}&name={urllib.parse.quote(name)}'


def update_db_name_de(item_id: int, name: str):
    __ts = time.perf_counter()
    db = sqlalchemy.create_engine('sqlite:///auctionator.db')
    session = sqlalchemy.orm.Session(db)
    Base.metadata.create_all(db)
    session.execute(sqlalchemy.text(f'UPDATE items SET name_de = "{name}" WHERE item_id = {item_id}'))
    logger = logging.getLogger('auctionator')
    logger.info(f'Updating the german name for item {item_id} took {time.perf_counter() - __ts: 0.2f}s.')


def test():
    write_data()
    get_item_prices()


if __name__ == '__main__':
    test()
