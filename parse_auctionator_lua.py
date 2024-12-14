import logging
import re
import time
import datetime
import os

from slpp import slpp as lua

import database_interactions as daint


def parse_data(content: str = None):
    __ts = time.perf_counter()
    logger = logging.getLogger('auctionator')

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

    # get the items from the database
    items = daint.get_items()

    # make dictionaries
    item_names = {name: ided for ided, names in items.items() for name in names if name}

    # make dicts to transform the names
    skipped_items = []
    for item, value_dict in history.items():
        # get the current id
        curr_id = int(value_dict['is'][0])

        # check whether we can find this item in our database
        if curr_id not in items:

            # keep track of skipped items
            error_str = f'Invalid Items in File - Id Unknown! Error with Item {item}, Id={curr_id}.'
            skipped_items.append((item, curr_id, error_str))

    # delete the corresponding item
    for item, _, _ in skipped_items:
        del history[item]

    # save how many we skipped from the pricing history
    skipped_pricing_history = len(skipped_items)
    pricing_history_items = len(history)

    # parse the recent scan
    scan_line = [line for line in content.splitlines() if line.startswith("AUCTIONATOR_LAST_SCAN_TIME")]
    assert len(scan_line) <= 1, 'Something with the .lua file is off.'
    scan_time = int(scan_line[0].split(' = ')[-1])

    # check that the line exists
    scan = []
    if scan_line:

        # check that we have the price database history
        scan = variable_dict['AUCTIONATOR_PRICE_DATABASE']

        # check that it is the right db verions
        if scan['__dbversion'] != 2:
            error_str = f'Invalid dbversion in this file! dbverions: {scan["__dbversion"]}.'
            logger.error(f'Logging error! --> {error_str}')
            return False, error_str

        # check that its is the right server
        if "Rising-Gods_Alliance" not in scan:
            error_str = f'Your lua file does not contain scan information from Rising Gods.'
            logger.error(f'Logging error! --> {error_str}')
            return False, error_str

        # extract the scan
        scan = scan["Rising-Gods_Alliance"]

        # check whether we find items with the names in the dict
        for name, value in scan.items():

            # continue if we have not yet found a name
            if name not in item_names:
                skipped_items.append((name, -1))
                continue

            # link the item name to an id
            item_id = item_names[name]

            # create a history entry for the current item
            if name not in history:
                history[name] = dict()
                history[name]['is'] = (item_id, 0)
            history[name][(scan_time, "")] = (int(value), 1)

    logger.info(f'Parsing took {time.perf_counter()-__ts: 0.2f}s. We skipped: {skipped_pricing_history}(missing id) + '
                f'{len(skipped_items)-skipped_pricing_history} (missing name).')

    # append the values to the skipped items
    skipped_items.append(("", (skipped_pricing_history, len(skipped_items)-skipped_pricing_history),
                          (pricing_history_items, len(scan))))
    return history, skipped_items


def convert_auctionator_time(time_int: int, zero_time: float):
    # https://github.com/alchem1ster/WotLK-Auctionator/blob/b4e3eb68dcc7cc1142b73d8a978d471bf0d4110a/Auctionator/Auctionator.lua#L4486
    return (time_int*60) + zero_time


def pretty_print_hist(hist: dict[tuple[int | str, str]: tuple[int, int]]):
    for (date, typed), (price, count) in hist.items():
        if date != 'i':
            print(datetime.datetime.fromtimestamp(date), typed, price, count)


def test():
    parse_data()


if __name__ == '__main__':
    test()
