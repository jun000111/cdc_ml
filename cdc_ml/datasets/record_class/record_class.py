def assign_class_type(df_records: pd.DataFrame, df_class: pd.DataFrame):
    """left join the records with the class type"""
    df_customer_records = df_records.merge(df_class, on="username", how="left")
    df_customer_records["is_one_team"] = df_customer_records["is_one_team"].fillna(0).astype(int)
    return df_customer_records
