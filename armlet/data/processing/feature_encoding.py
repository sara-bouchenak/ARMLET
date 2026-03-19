import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from armlet.data.processing.utils import apply_processing_to_data, merge_all_X_data


def one_hot_encoding_pipeline(data):

    df_X = merge_all_X_data(data)

    categorical_cols = df_X.select_dtypes(include=["object"]).columns.tolist()

    if categorical_cols != []:
    
        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.fit(df_X[categorical_cols])

        one_hot_data = apply_processing_to_data(
            data=data, 
            function=transform_data_with_one_hot_encoding,
            one_hot_encoder=one_hot_encoder,
            categorical_cols=categorical_cols,
        )
        return one_hot_data

    else:
        return data

def transform_data_with_one_hot_encoding(subdata, data_key, one_hot_encoder, categorical_cols):
    X, y = subdata
    one_hot_array = one_hot_encoder.transform(X[categorical_cols]).toarray()
    one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_encoder.get_feature_names_out())
    X_one_hot = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True), one_hot_df], axis=1)
    return X_one_hot, y
