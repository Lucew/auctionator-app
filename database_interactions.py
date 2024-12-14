import os
import time
import datetime
import collections
import logging

import sqlalchemy
import pygsheets
import pandas as pd
import urllib.parse

import database_definition as dbd
import parse_auctionator_lua as pal


def get_db_engine():
    db = sqlalchemy.create_engine('sqlite:///auctionator.db')
    dbd.Base.metadata.create_all(db)
    return db


def write_to_database(hist: dict[tuple[int | str, str]: tuple[int, int]]):
    __ts = time.perf_counter()
    logger = logging.getLogger('auctionator')

    with sqlalchemy.orm.Session(get_db_engine()) as session:
        # go through the data and create items if necessary, otherwise check that id is unique
        for _, value_dict in hist.items():

            # get the current id
            curr_id = int(value_dict['is'][0])

            # go through the prices and write them to the database
            for typed, stacks in value_dict.items():

                # skip the id
                if typed == 'is':
                    continue

                # skip all mean values
                if typed[1]:
                    continue

                # check whether this item with this date is already there
                check_query = session.query(dbd.Price).filter_by(id=curr_id, unix_timestamp=typed[0]).one_or_none()
                if check_query is None:
                    session.add(dbd.Price(id=curr_id, unix_timestamp=typed[0], price=stacks[0], stacks=int(stacks[1])))
        session.commit()
    logger.info(f'Writing to DB took {time.perf_counter() - __ts: 0.2f}s')
    return True, 'Success!'


def get_item_prices():
    __ts = time.perf_counter()

    with sqlalchemy.orm.Session(get_db_engine()) as session:

        # get all the prices for an item and order them by the time
        grouped_prices = session.query(dbd.Price, dbd.Item).filter(dbd.Price.id == dbd.Item.item_id)\
            .order_by(dbd.Price.unix_timestamp).all()

        # write the prices in to a dict
        prices_dict = collections.defaultdict(list)
        for group in grouped_prices:
            prices_dict[(group[1].item_id, group[1].name_de, group[1].name)].append(
                (group[0].unix_timestamp, group[0].price/group[0].stacks))

    logger = logging.getLogger('auctionator')
    logger.info(f'Querying the DB for prices took {time.perf_counter() - __ts: 0.2f}s')
    return prices_dict


def get_items():
    __ts = time.perf_counter()

    with sqlalchemy.orm.Session(get_db_engine()) as session:
        # query all items
        result = session.execute(sqlalchemy.select(dbd.Item.item_id, dbd.Item.name, dbd.Item.name_de)).all()
        item_dict = {rs.item_id: (rs.name, rs.name_de) for rs in result}

    # logg the time it took
    logger = logging.getLogger('auctionator')
    logger.info(f'Querying the DB for all items took {time.perf_counter() - __ts: 0.2f}s')
    return item_dict


def get_spells():
    __ts = time.perf_counter()

    with sqlalchemy.orm.Session(get_db_engine()) as session:

        # search through spell names
        result = session.query(dbd.SpellNames).all()

        # collect the results
        spell_dict = {spell.id: (spell.name_en, spell.name_de, spell.cooldown, spell.profession_name, spell.skill)
                      for spell in result}

        # collect the profession
        professions = collections.defaultdict(list)
        for spell in result:
            # check that profession is not empty
            if spell.profession_name:
                professions[spell.profession_name].append(spell.id)

    # logg the time it took
    logger = logging.getLogger('auctionator')
    logger.info(f'Querying the DB for all spells took {time.perf_counter() - __ts: 0.2f}s')
    return spell_dict, professions


def get_most_recent_item_price():
    __ts = time.perf_counter()

    # create the sql statement
    sql_statement = 'SELECT i.item_id, p.price FROM prices p JOIN items i ON p.id == i.item_id ' \
                    'WHERE p.unix_timestamp == (SELECT MIN(unix_timestamp) FROM prices p2 WHERE p2.id == p.id)'

    # get the session
    with sqlalchemy.orm.Session(get_db_engine()) as session:
        recent_price = session.execute(sqlalchemy.text(sql_statement)).all()
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
        history, skipped_items = pal.parse_data(input_str)
    except ValueError as e:
        logger.error(f'Parsing error: ValueError --> {str(e)}.')
        return False, 'Invalid Lua file.', []
    except AssertionError as e:
        logger.error(f'Parsing error: AssertionError --> {str(e)}.')
        return False, 'Invalid Lua file.', []
    write_success = write_to_database(history)
    # write_to_gsheet(prices_dict)
    return *write_success, skipped_items


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
    df_dict = {'Name': [], 'Name (en)': [], 'Price': [], 'Date': [], 'Gold': [], 'Id': []}
    name2link = dict()
    for ((idd, name, name_de), sorted_prices) in history_dict.items():
        name_de = name_de if name_de else name
        df_dict['Name (en)'].extend(name for _ in range(len(sorted_prices)))
        df_dict['Name'].extend(name_de for _ in range(len(sorted_prices)))
        df_dict['Price'].extend(price for _, price in sorted_prices)
        df_dict['Date'].extend(date for date, _ in sorted_prices)
        df_dict['Gold'].extend(price2gold(price) for _, price in sorted_prices)
        df_dict['Id'].extend(idd for _ in range(len(sorted_prices)))
        name2link[name_de] = name2dblink(idd, name_de)
    df = pd.DataFrame(df_dict)
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    return df, name2link


def name2dblink(item_id: int, name: str):
    return f'https://db.rising-gods.de/?item={item_id}&name={urllib.parse.quote(name)}'


def update_db_item_name_de(item_id: int, name: str):
    __ts = time.perf_counter()

    # get the session to the database
    with sqlalchemy.orm.Session(get_db_engine()) as session:
        # update the item
        query_text = sqlalchemy.text('UPDATE items SET name_de = :name_de WHERE item_id = :item_id')
        session.execute(query_text, {'name_de': name, 'item_id': item_id})
        session.commit()

    # logg the time and execution
    logger = logging.getLogger('auctionator')
    logger.info(f'Updating the german name for item {item_id} took {time.perf_counter() - __ts: 0.2f}s.')


def insert_db_spell_name(spell_id: int, name_de: str, name_en: str, profession_name: str, profession_name_en: str,
                         cooldown: int, skill_level: int):
    __ts = time.perf_counter()

    # make the query string
    query_stmt = 'INSERT INTO spell_names VALUES ' \
                 '(:spell_id, :profession_name, :profession_name_en, :cooldown, :skill_level, :name_de, :name_en, "")'

    # get the session to the database
    with sqlalchemy.orm.Session(get_db_engine()) as session:
        query = session.query(dbd.SpellNames).filter(dbd.SpellNames.id == spell_id).one_or_none()

        if query is None:
            # update the item
            values = {"spell_id": spell_id, "profession_name": profession_name,
                      "profession_name_en": profession_name_en, "cooldown": cooldown, "skill_level": skill_level,
                      "name_de": name_de, "name_en": name_en}
            session.execute(sqlalchemy.text(query_stmt), values)
            session.commit()

        # logg the time and execution
        logger = logging.getLogger('auctionator')
        logger.info(f'Updating the german name for spell {spell_id} took {time.perf_counter() - __ts: 0.2f}s.')


def clean_item_database():

    # delete all items that have expansion four
    # or have an item with a larger expansion
    statement = "DELETE FROM items AS it1 WHERE it1.expansion_id == 4 OR " \
                "(SELECT COUNT(*) FROM items AS it2 WHERE it2.item_id == it1.item_id " \
                "and it2.expansion_id > it1.expansion_id) == 1"

    # get the session to the database
    with sqlalchemy.orm.Session(get_db_engine()) as session:
        # execute the statement
        session.execute(sqlalchemy.text(statement))

        # delete the spell names table
        # session.execute(sqlalchemy.text('DROP TABLE spell_names'))
    session.commit()


def test():
    clean_item_database()
    # write_data()
    # get_item_prices()


if __name__ == '__main__':
    test()
