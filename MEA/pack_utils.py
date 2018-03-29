import pandas as pd
import numpy as np


LIST_OF_CSVS = ['./1.09/MEA_Patch_1_09.csv', './1.10/MEA_Patch_1_10.csv']
COMMENT_COLUMNS = ['Date', 'Notes', 'Variant present', 'Ultra Rares']
PM_EXCLUDE_PACK_TYPES = ['Apex Elite', 'Jumbo Premium', 'Celebratory']
PM_PACK_TYPES = ['Reserves', 'Advanced', 'Jumbo Supply', 'Premium',
                 'Arsenal', 'Expert', 'Basic', 'Ammo Priming', 'Supply',
                 'Technical Mods']
AP_CREDS_CONV = 100000.0 / 300.0
SUPPORT_PACK_TYPES = ['Jumbo Supply', 'Ammo Priming', 'Supply']
BASIC_PACK_TYPES = ['Basic']
UR_PACK_TYPES = ['Reserves', 'Premium', 'Arsenal', 'Expert']


def make_basic_dataframe(list_of_csvfiles):
    first_file = list_of_csvfiles[0]
    packs_raw = pd.read_csv(first_file)
    for f in list_of_csvfiles[1:]:
        packs_raw = packs_raw.append(pd.read_csv(f))
    return packs_raw


def clean_nas_by_field(df, field='Price'):
    df = df[pd.notnull(df[field])]
    return df


def drop_comment_columns(df, list_of_columns=COMMENT_COLUMNS):
    for col in list_of_columns:
        try:
            df = df.drop(columns=[col])
        except ValueError:
            pass
    return df


def make_pricing_model_starter(list_of_csvfiles, purch_type='Credits'):
    df = make_basic_dataframe(list_of_csvfiles)
    df = clean_nas_by_field(df)
    df = drop_comment_columns(df)
    exclude_list = PM_EXCLUDE_PACK_TYPES + BASIC_PACK_TYPES
    df = df[~df['Pack'].isin(exclude_list)]
    df = df[df['AP/Creds'] == purch_type]
    # df['Price'][df['AP/Creds'] == 'AP'] *= AP_CREDS_CONV
    return df


def get_support_packs_df(df):
    new_df = df.copy()
    new_df = new_df[new_df['Pack'].isin(SUPPORT_PACK_TYPES)]
    return new_df


def get_ur_packs_df(df):
    new_df = df.copy()
    new_df = new_df[new_df['Pack'].isin(UR_PACK_TYPES)]
    return new_df


def get_ur_counts(ur_df):
    n_ur_char = np.sum(ur_df['UR Char.'])
    n_ur_weap = np.sum(ur_df['UR Weap.'])
    return n_ur_char, n_ur_weap


def convert_ap_to_creds(df):
    new_df = df.copy()
    new_df['Price'][new_df['AP/Creds'] == 'AP'] *= AP_CREDS_CONV
    return new_df
    

def test_one_by_one():
    print('Basic packs:')
    df = make_basic_dataframe(LIST_OF_CSVS)
    print(df.info())
    print('Clean NAs for Price')
    df = clean_nas_by_field(df)
    print(df.info())
    print('Drop comment columns')
    df = drop_comment_columns(df)
    print(df.info())


def test_starter():
    print('Pricing model starter')
    df = make_pricing_model_starter(LIST_OF_CSVS)
    print(df.info())


if __name__ == '__main__':
    test_starter()
