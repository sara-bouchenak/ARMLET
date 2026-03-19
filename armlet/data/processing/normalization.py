import pandas as pd

from sklearn.preprocessing import StandardScaler

from armlet.data.processing.utils import apply_processing_to_data


def normalization_pipeline(data, cols_to_exclude: list[str] = []):

    list_df = [X for (X, y) in data["clients_train"].values()]
    df = pd.concat(list_df, axis=0, ignore_index=True)

    norm_cols = df.columns.tolist()
    norm_cols = [col for col in norm_cols if col not in cols_to_exclude]

    if norm_cols != []:

        scaler = StandardScaler()
        scaler.fit(df[norm_cols])

        normalized_data = apply_processing_to_data(
            data=data,
            function=transform_data_with_scaler,
            scaler=scaler,
            norm_cols=norm_cols,
        )
        return normalized_data

    else:
        return data

def transform_data_with_scaler(subdata, data_key, scaler, norm_cols):
    X, y = subdata
    X[norm_cols] = scaler.transform(X[norm_cols])
    return X, y
