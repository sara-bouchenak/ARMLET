import pandas as pd
from functools import partial


def apply_processing_to_data(data, function, **func_args):

    if func_args:
        function = partial(
            function,
            **func_args,
        )

    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    processed_client_data = function(client_data, key)
                    data[key][id_client] = processed_client_data
        else:
            if subdata != None:
                processed_server_data = function(subdata, key)
                data[key] = processed_server_data
    return data

def merge_all_X_data(data) -> pd.DataFrame:
    list_df = []
    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    list_df.append(X)
        else:
            if subdata != None:
                X, y = subdata
                list_df.append(X)
    df_X = pd.concat(list_df, axis=0, ignore_index=True)
    return df_X
